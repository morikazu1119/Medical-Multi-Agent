import argparse
import base64
import os
from io import BytesIO
from typing import Optional

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel

# ─── 環境変数ロード & Gemini クライアント設定 ─────────────────────────
load_dotenv("key.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません（key.env を確認してください）")
genai.configure(api_key=GEMINI_API_KEY)

# ─── Gemini マルチモーダル Flash モデル初期化 ─────────────────────────
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# ─── FastMCP インスタンス作成 ─────────────────────────────
mcp = FastMCP("gemini-diagnosis-mcp")


# ─── Pydantic モデル定義（image を Optional に） ────────────────────────
class MCPRequest(BaseModel):
    image: Optional[str] = None  # base64 エンコード済み画像（任意）
    symptom: str  # 症状テキスト（必須）


class MCPResponse(BaseModel):
    diagnosis: str  # Gemini が返す診断テキスト


# ─── MCP ツール定義 ─────────────────────────────────────────────
@mcp.tool()
def diagnose(request: MCPRequest) -> MCPResponse:
    parts = []

    # 画像があればデコード＆PNG 再エンコードして parts に追加
    if request.image:
        try:
            img_bytes = base64.b64decode(request.image)
            img = Image.open(BytesIO(img_bytes))
        except Exception as e:
            raise RuntimeError(f"画像デコード失敗: {e}")

        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({"mime_type": "image/png", "data": img_b64_png})

    # 症状テキストは必ず最後に追加
    prompt = (
        f"{request.symptom} 画像と症状に基づいて考えられる病気と" "推奨される処置を日本語で教えてください。"
        if request.image
        else f"{request.symptom} のみをもとに考えられる病気と推奨される処置を日本語で教えてください。"
    )
    parts.append(prompt)

    # Gemini 呼び出し
    try:
        resp = chat.send_message(parts)
        diag_text = resp.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini 呼び出しエラー: {e}")

    return MCPResponse(diagnosis=diag_text)


# ─── サーバ起動定義 ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gemini FastMCP Server")
    parser.add_argument(
        "--host", type=str, required=True, help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port to bind the server"
    )
    args = parser.parse_args()

    uvicorn.run(mcp.streamable_http_app, host=args.host, port=int(args.port), workers=1)


if __name__ == "__main__":
    main()
