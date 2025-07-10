import asyncio
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
# 設定：ログ出力レベルの設定
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 環境変数のロードとAPIキー取得
# -----------------------------------------------------------------------------
load_dotenv("key.env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY が設定されていません。環境変数 key.env を確認してください。")
    raise RuntimeError("GEMINI_API_KEY が設定されていません。")

# Gemini クライアントの初期化
client = genai.Client(api_key=API_KEY)


# -----------------------------------------------------------------------------
# 関数：MCP サーバーへ問い合わせ
# -----------------------------------------------------------------------------
async def fetch_from_mcp(url: str, prompt: str) -> str:
    """
    指定の MCP URL にプロンプトを送り、テキスト応答を得て返却する。
    エラー時はエラーメッセージを返却。
    """
    logger.debug(f"MCP 呼び出し開始: {url}")
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
                logger.debug(f"MCP 応答受信: {url}")
                return response.text
    except Exception as e:
        err_msg = f"[ERROR] {url}: {e}"
        logger.error(err_msg)
        return err_msg


# -----------------------------------------------------------------------------
# 関数：複数結果の要約生成
# -----------------------------------------------------------------------------
async def summarize_results(responses: list[str]) -> str:
    """
    各 MCP の応答を結合し、Gemini API で日本語の要約を生成して返却。
    """
    combined_text = "\n\n".join(responses)
    summary_prompt = "以下は複数のMCPサーバーからの応答です。日本語で簡潔に要約してください。\n\n" + combined_text
    logger.info("要約プロンプトを生成し、Gemini に送信します。")
    summary = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=summary_prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    logger.info("要約応答を受信しました。")
    return summary.text


# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
async def main():
    # 1) 設定ファイルから MCP URL リストを取得
    config = OmegaConf.load("src/config/config.yaml")
    mcp_urls = config.pipeline.mcp_urls or []
    if not mcp_urls:
        logger.error("config.yaml に mcp_urls が設定されていません。")
        raise RuntimeError("src/config/config.yaml に mcp_urls が設定されていません。")

    # 2) 共通プロンプトの定義
    user_prompt = "頭が痛いです。何の病気の可能性がありますか？"

    # 3) 並列で各 MCP サーバーに問い合わせ
    logger.info(f"{len(mcp_urls)} 件のMCPサーバーに非同期リクエストを送信します。")
    tasks = [fetch_from_mcp(url, user_prompt) for url in mcp_urls]
    individual_results = await asyncio.gather(*tasks)

    # 4) 応答をまとめて要約
    summary = await summarize_results(individual_results)

    # 5) コンソール出力
    separator = "-" * 40
    print(separator)
    print("個別応答：")
    print(separator)
    for url, resp in zip(mcp_urls, individual_results):
        print(f"[{url}]\n{resp}\n")
    print(separator)
    print("要約：")
    print(separator)
    print(summary)

    # 関数返却値（必要に応じて他モジュールから利用可）
    return {
        "individual_responses": individual_results,
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# エントリポイント
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        logger.info("処理が完了しました。")
    except Exception as e:
        logger.exception(f"実行中に例外が発生しました: {e}")
        raise
