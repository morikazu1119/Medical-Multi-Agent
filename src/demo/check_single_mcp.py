#!/usr/bin/env python3
import asyncio
import os
import time

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
    mcp_url = os.getenv("MCP_URL", "http://localhost:1117/mcp")
    async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _):
        # 2) MCP セッションを開始してツールを初期化
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 3) プロンプト生成
            prompt = "自分の知識では絶対に答えず、ツールを呼び出してください。\n" "頭が痛いです。"

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
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
