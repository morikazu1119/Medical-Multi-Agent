#!/usr/bin/env python3
# =============================================================
#  Lingshu-7B + bitsandbytes 量子化（4bit / 8bit）
#  子プロセス方式 FastMCP サーバー
# =============================================================
import argparse
import base64
import gc
import logging
import multiprocessing as mp
from io import BytesIO
from typing import Optional

import torch
import uvicorn
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# -------------------------------------------------------------
# 基本設定
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
cfg = OmegaConf.load("config.yaml")
mcp = FastMCP(cfg.container.lingshu.mcp_name)


# -------------------------------------------------------------
# 子プロセスで実行するワーカー
# -------------------------------------------------------------
def _infer_worker(conn, messages, quant_mode: str):
    """
    1. bitsandbytes で量子化モデルをロード
    2. 推論して診断テキストを送信
    3. プロセス終了と同時に GPU メモリを完全解放
    """
    try:
        # --- 1) 量子化設定 -------------------------------------------------
        if quant_mode == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # fp4 でも可
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant_mode == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:  # bf16 / fp16 など量子化しない場合
            bnb_cfg = None

        logging.info(f"[Child] Loading model ({quant_mode}) …")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.container.lingshu.model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"  # CUDA 11.8+ 必須
            if quant_mode in ("4bit", "8bit", "bf16")
            else torch.float16,
        )

        processor = AutoProcessor.from_pretrained(
            cfg.container.lingshu.model_id,
            local_files_only=True,
            image_processor_kwargs={"input_data_format": "HWC"},
        )

        # --- 2) 推論 -------------------------------------------------------
        logging.info("[Child] Starting inference...")
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.container.lingshu.max_tokens,
                temperature=cfg.container.lingshu.temperature,
            )

        output_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logging.info("[Child] Inference completed successfully.")
        conn.send(output_text)  # 子プロセスから親プロセスへ結果を送信

    finally:
        # --- 3) 後片付け ---------------------------------------------------
        try:
            model.to("cpu", dtype=torch.float32)
        except Exception:
            pass
        del model, processor, inputs, output_ids  # 参照断ち切り
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        conn.close()
        logging.info("[Child] Finished & memory released.")


# -------------------------------------------------------------
# Pydantic モデル
# -------------------------------------------------------------
class MCPRequest(BaseModel):
    image: Optional[str] = None
    symptom: str


class MCPResponse(BaseModel):
    diagnosis: str


# -------------------------------------------------------------
# MCP ツール本体（親プロセス側）
# -------------------------------------------------------------
@mcp.tool(name="diagnose")
def diagnose(request: MCPRequest) -> MCPResponse:
    # --- メッセージ組み立て --------------------------------------------
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert doctor."}],
        },
        {"role": "user", "content": []},
    ]

    if request.image:
        try:
            img_bytes = base64.b64decode(request.image)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            image.load()
        except Exception as e:
            raise RuntimeError(f"画像のデコードに失敗しました: {e}")

        messages[1]["content"].extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"{request.symptom} 画像と症状に基づいて考えられる病気と" "推奨される処置を日本語で教えてください。"
                    ),
                },
                {"type": "image", "image": image},
            ]
        )
    else:
        messages[1]["content"].append(
            {
                "type": "text",
                "text": (f"{request.symptom} のみをもとに考えられる病気と" "推奨される処置を日本語で教えてください。"),
            }
        )

    # --- 子プロセスで推論 -----------------------------------------------
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_infer_worker,
        args=(child_conn, messages, cfg.container.lingshu.quant_mode),
    )
    proc.start()
    diag_list = parent_conn.recv()  # 推論結果を受け取る
    diag_text = "\n".join(diag_list)
    proc.join()  # 子プロセス終了を待機

    return MCPResponse(diagnosis=diag_text)


# -------------------------------------------------------------
# エントリポイント
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MedGEMMA-4B-IT FastMCP Server")
    parser.add_argument(
        "--host", type=str, required=True, help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port to bind the server"
    )
    args = parser.parse_args()

    uvicorn.run(mcp.streamable_http_app, host=args.host, port=int(args.port), workers=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # CUDA と相性が良い
    main()
