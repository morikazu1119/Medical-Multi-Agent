#!/usr/bin/env python3
import asyncio
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from omegaconf import OmegaConf

# 環境変数のロード
load_dotenv("key.env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY が設定されていません。")

# Gemini クライアントの初期化
client = genai.Client(api_key=API_KEY)


async def fetch_from_mcp(url: str, prompt: str) -> str:
    """
    指定した MCP URL に対してプロンプトを送信し、レスポンスのテキストを返す。
    エラーが発生した場合はエラーメッセージを返す。
    """
    try:
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        tools=[session],
                    ),
                )
                return response.text
    except Exception as e:
        return f"[ERROR] {url}: {e}"


async def summarize_results(results: list[str]) -> str:
    """
    複数のMCP応答をまとめて要約を生成する。
    """
    combined = "\n\n".join(results)
    summary_prompt = "以下の複数のMCPサーバーからの応答結果をまとめて、日本語で簡潔に要約してください。\n\n" + combined
    summary_response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=summary_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )
    return summary_response.text


async def main():
    # 1) src/config/config.yaml から複数の MCP URL を読み込む
    conf = OmegaConf.load("src/config/config.yaml")
    mcp_urls = conf.pipeline.mcp_urls
    if not mcp_urls:
        raise RuntimeError("src/config/config.yaml に mcp_urls が設定されていません。")

    # 2) プロンプト生成
    prompt = "頭が痛いです。何の病気の可能性がありますか？"

    # 3) 並列で各 MCP に問い合わせ
    tasks = [fetch_from_mcp(url, prompt) for url in mcp_urls]
    individual_results = await asyncio.gather(*tasks)

    # 4) 結果を要約
    summary = await summarize_results(individual_results)

    # 5) 結果表示および返却
    print("---------------")
    print("--- 個別応答 ---")
    print("---------------")
    for url, res in zip(mcp_urls, individual_results):
        print(f"MCP URL: {url}\n{res}\n")
    print("---------------")
    print("----- 要約 -----")
    print("---------------")
    print(summary)

    return {
        "individual_responses": individual_results,
        "summary": summary,
    }


if __name__ == "__main__":
    # 実行時間計測
    start_time = time.time()
    result = asyncio.run(main())
    end_time = time.time()
    print(f"実行時間: {end_time - start_time:.2f}秒")
