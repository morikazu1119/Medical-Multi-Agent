import asyncio
import logging
import os

from dotenv import load_dotenv
from google import genai
from omegaconf import OmegaConf

from src.mcp_clients.gemini.functions import call_mcp, summarize

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 環境変数のロード
load_dotenv("key.env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY is not set.")
    raise RuntimeError("GEMINI_API_KEY is not set.")

# Gemini クライアント初期化
client = genai.Client(api_key=API_KEY)

PROMPT = "頭が痛いです。何の病気の可能性がありますか？"


async def main():
    # 設定ファイルの読み込み
    config = OmegaConf.load("src/config/config.yaml")
    mcp_urls = config.pipeline.mcp_urls or []
    if not mcp_urls:
        raise RuntimeError("MCP URL が設定されていません。")
    logger.info(f"{len(mcp_urls)} mcp urls found.")

    tasks = [call_mcp(url, PROMPT, client) for url in mcp_urls]
    responses = await asyncio.gather(*tasks)

    summary = await summarize(responses, client)

    separator = "-" * 40
    print(separator)
    print("Single Response：")
    print(separator)
    for url, resp in zip(mcp_urls, responses):
        print(f"[{url}]\n{resp}\n")
    print(separator)
    print("Summarization：")
    print(separator)
    print(summary)

    return {
        "individual_responses": responses,
        "summary": summary,
    }


if __name__ == "__main__":
    try:
        asyncio.run(main())
        logger.info("Process completed successfully.")
    except Exception as e:
        logger.exception(f"Process is failed: {e}")
