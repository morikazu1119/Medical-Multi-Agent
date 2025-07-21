#!/usr/bin/env python3
# =============================================================
# MediPhi-it (text) + MedSigLIP-448 (image) FastMCP Server
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

# ──────────────────────────────────────────────────────────────
# 設定ロード
# ──────────────────────────────────────────────────────────────
cfg = OmegaConf.load("config.yaml")
mediphi_cfg = cfg.container.mediphi
blip_cfg = cfg.container.blip
mcp = FastMCP(mediphi_cfg.mcp_name)


# ──────────────────────────────────────────────────────────────
# 子プロセスワーカー
# ──────────────────────────────────────────────────────────────
def _infer_worker(
    conn, symptom: str, image: Image, mediphi_cfg: OmegaConf, blip_cfg: OmegaConf
):
    try:
        if image is not None:
            # --- 画像特徴抽出: BLIP -----------
            logging.info("[Child] Loading BLIP model ...")
            processor = BlipProcessor.from_pretrained(
                blip_cfg.model_id,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=True,
            )
            model = BlipForConditionalGeneration.from_pretrained(
                blip_cfg.model_id,
                local_files_only=True,
                device_map="auto",
            ).to("cuda")

            logging.info("[Child] Generating image caption ...")
            # conditional image captioning
            text_list = [
                "Modality: ",  # 撮像法をここに返してくれる
                "Organ: ",  # 臓器名をここに返してくれる
                "Findings: ",  # 検出項目をここに返してくれる
            ]
            output_text_list = []
            for text in text_list:
                inputs = processor(image, text, return_tensors="pt").to("cuda")
                out = model.generate(
                    **inputs,
                    early_stopping=True,
                    max_new_tokens=blip_cfg.max_tokens,  # 生成するトークン数
                    do_sample=False,  # サンプリングを無効化
                    repetition_penalty=1.2,  # 繰り返しペナルティ
                    num_beams=4,  # ビームサーチのビーム数
                )
                output_text = processor.decode(out[0], skip_special_tokens=True)
                output_text_list.append(output_text)
                print(f"output_text: {text + output_text}")

            image_caption = " ".join(output_text_list)
            logging.info(f"[Child] Image caption: {image_caption}")
        else:
            image_caption = None
            logging.info("[Child] No image provided, skipping captioning.")

        # --- MediPhi 量子化設定 ------------------------------
        qm = mediphi_cfg.quant_mode
        if qm == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif qm == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_cfg = None

        logging.info("[Child] Loading MediPhi-it …")
        mediphi_model = AutoModelForCausalLM.from_pretrained(
            mediphi_cfg.model_id,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        mediphi_tokenizer = AutoTokenizer.from_pretrained(
            mediphi_cfg.model_id, local_files_only=True, use_fast=True
        )

        # --- プロンプト生成 & 推論 ---------------------------
        user_prompt = (
            "You are a medical assistant. Provide differential diagnoses (not final diagnoses), "
            "list key medical keywords extracted, and suggest next steps.\n"
            f'Image finding (textual surrogate): "{image_caption}"\n'
            f"Symptoms: {symptom}\n"
            "Output sections: 1) Keywords 2) Differential Dx 3) Recommended Next Steps.\n"
            "Use concise medical English terms."
        )

        mediphi_pipeline = pipeline(
            task="text-generation",
            model=mediphi_model,
            tokenizer=mediphi_tokenizer,
        )

        logging.info("[Child] Generating diagnosis ...")
        outputs = mediphi_pipeline(
            user_prompt,
            max_new_tokens=mediphi_cfg.max_tokens,
            temperature=mediphi_cfg.temperature,
            do_sample=False,
            return_full_text=False,
        )
        diagnosis = outputs[0]["generated_text"]
        conn.send(diagnosis)
    except Exception as e:
        print(f"[Child] Error: {e}")
        conn.send(f"Error: {e}")

    finally:
        # メモリ解放
        try:
            model.to("cpu", dtype=torch.float32)
        except Exception:
            pass
        del mediphi_model, mediphi_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────────────────────
class MCPRequest(BaseModel):
    image: Optional[str] = None
    symptom: str


class MCPResponse(BaseModel):
    diagnosis: str


# ──────────────────────────────────────────────────────────────
# MCP ツール
# ──────────────────────────────────────────────────────────────
@mcp.tool(name="diagnose")
def diagnose(request: MCPRequest) -> MCPResponse:
    if request.image:
        try:
            img_bytes = base64.b64decode(request.image)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            image.load()
        except Exception as e:
            raise RuntimeError(f"画像のデコードに失敗しました: {e}")
    else:
        image = None
    parent_conn, child_conn = mp.Pipe(False)
    proc = mp.Process(
        target=_infer_worker,
        args=(child_conn, request.symptom, image, mediphi_cfg, blip_cfg),
    )
    proc.start()
    diagnosis = parent_conn.recv()
    proc.join()
    return MCPResponse(diagnosis=diagnosis)


# ──────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Blip + Mediphi FastMCP Server")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    uvicorn.run(mcp.streamable_http_app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
