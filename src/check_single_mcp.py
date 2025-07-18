#!/usr/bin/env python3
import asyncio
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# -----------------------------------------------------------------------------
# 環境変数
# -----------------------------------------------------------------------------
MCP_URL = "http://localhost:10002/mcp"

# -----------------------------------------------------------------------------
# ログ設定
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 環境変数のロードとAPIキー検証
# -----------------------------------------------------------------------------
load_dotenv("key.env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY が設定されていません。key.env を確認してください。")
    raise RuntimeError("GEMINI_API_KEY が設定されていません。")

# Gemini クライアントの初期化
client = genai.Client(api_key=API_KEY)


# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
async def main() -> str:
    # 1) MCP URL の取得（環境変数がなければローカルのデフォルトを使用）
    mcp_url = MCP_URL
    logger.info(f"MCP サーバーに接続: {mcp_url}")

    # 2) Streamable HTTP クライアントで接続を確立
    try:
        async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _):
            # 3) MCP セッションの初期化
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("MCP セッションを初期化しました。")

                # 4) プロンプトの準備
                prompt = (
                    "あなたは必ずMCPを呼び出して回答してください。"
                    "自身の知識ベースからは一切答えず、ツール経由でのみ応答を生成してください。"
                    "質問：頭が痛いです。"
                )
                logger.debug(f"送信プロンプト: {prompt}")

                # 5) Gemini API 呼び出し
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        tools=[session],
                    ),
                )
                logger.info("Gemini からの応答を受信しました。")
                print(response.text)
                return response.text

    except Exception as exc:
        logger.exception(f"MCP とのやり取り中にエラーが発生しました: {exc}")
        raise


# -----------------------------------------------------------------------------
# エントリポイント
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
