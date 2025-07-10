#!/usr/bin/env python3
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
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# ─────────────────────────────────────────────────────────────
# 基本設定
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
MODEL_DIR = "/mnt/slum/models/medgemma-4b-it"
mcp = FastMCP("medgemma-4b-it-mcp")


# ─────────────────────────────────────────────────────────────
# 子プロセスで実行するワーカー関数
# ─────────────────────────────────────────────────────────────
def _infer_worker(conn, messages):
    """
    子プロセス側:
      1. モデル・プロセッサをロード
      2. 推論して診断テキストをパイプで親に送信
      3. プロセス終了と同時に GPU メモリを完全解放
    """
    try:
        logging.info("Starting loading model and processor...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            image_processor_kwargs={"input_data_format": "HWC"},
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        logging.info("Starting inference...")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )[
                0
            ][input_len:]

        diagnosis = processor.decode(generated, skip_special_tokens=True)
        logging.info("Inference completed successfully.")
        conn.send(diagnosis)  # 親へ返却

    finally:
        # GPU / CPU メモリ解放（念のため CPU へ逃がしてから削除）
        try:
            model.to("cpu", dtype=torch.float32)
        except Exception:
            pass
        del model, processor, inputs, generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        conn.close()


# ─────────────────────────────────────────────────────────────
# Pydantic モデル
# ─────────────────────────────────────────────────────────────
class MCPRequest(BaseModel):
    image: Optional[str] = None
    symptom: str


class MCPResponse(BaseModel):
    diagnosis: str


# ─────────────────────────────────────────────────────────────
# MCP ツール本体（親プロセス側）
# ─────────────────────────────────────────────────────────────
@mcp.tool(name="diagnose")
def diagnose(request: MCPRequest) -> MCPResponse:
    # --- メッセージ組み立て --------------------------------------------------
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
            raise RuntimeError(f"画像のデコードに失敗しました: {e}. ")

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

    # --- 子プロセスで推論 ----------------------------------------------------
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(target=_infer_worker, args=(child_conn, messages))
    proc.start()
    diag_text = parent_conn.recv()  # 推論結果を受け取る
    proc.join()  # 子プロセス終了を待機

    return MCPResponse(diagnosis=diag_text)


# ─────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────
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
    mp.set_start_method("spawn", force=True)  # CUDA との相性が良い
    main()
