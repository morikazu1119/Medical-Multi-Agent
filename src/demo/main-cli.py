import asyncio
import os

from dotenv import load_dotenv
from google import genai
from omegaconf import OmegaConf

from src.mcp_clients.gemini.functions import run_discussion  # 既存の関数を流用

PROMPT = "頭が痛いです。何の病気の可能性がありますか？"
MAX_TURNS = 5


def main():
    # 環境変数のロードと APIキー確認
    load_dotenv("key.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。key.env を確認してください。")
        return

    client = genai.Client(api_key=api_key)

    # 設定ファイル読み込み
    config_path = "src/config/config.yaml"
    try:
        config = OmegaConf.load(config_path)
        mcp_urls = config.pipeline.mcp_urls
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        return

    if not mcp_urls:
        print("エラー: config.yaml に mcp_urls が定義されていません。")
        return

    # 実行
    print("\n==== 議論を開始します ====\n")
    history, summary = asyncio.run(run_discussion(PROMPT, mcp_urls, MAX_TURNS, client))

    # 出力
    for turn in history:
        print(f"\n--- Turn {turn['turn']} ---")
        print(f"Prompt: {turn['prompt']}")
        for res in turn["results"]:
            print(f"[{res['url']}]\n{res['response']}\n")

    print("\n" + "=" * 40)
    print("最終要約レポート")
    print("=" * 40)
    print(summary)


if __name__ == "__main__":
    main()
