#!/usr/bin/env python3
import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv("key.env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY が設定されていません。")

# Gemini クライアントの初期化
client = genai.Client(api_key=API_KEY)


async def main():
    # 1) Streamable HTTP で MCP サーバーに接続
    mcp_url = os.getenv("MCP_URL", "http://localhost:1111/mcp")
    async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _):
        # 2) MCP セッションを開始してツールを初期化
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 3) プロンプト生成
            prompt = "おすすめの料理を教えてください。\n"

            # 4) Gemini に問い合わせ（tools に session を渡す）
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # ここで MCP セッションを指定
                ),
            )

            # 5) 結果表示
            print(response.text)

            return response.text


if __name__ == "__main__":
    asyncio.run(main())
