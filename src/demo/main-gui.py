import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from omegaconf import OmegaConf

from src.mcp_clients.gemini.functions import run_discussion

# 環境変数ロードとAPIキー取得
load_dotenv("key.env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY が設定されていません。")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Streamlit UI
st.title("MCP 専門家議論 - GUI インターフェース")

if "mcp_urls" not in st.session_state:
    st.session_state["mcp_urls"] = []

config_path = st.text_input("設定ファイルパス", value="src/config/config.yaml")
if st.button("設定読み込み"):
    try:
        conf = OmegaConf.load(config_path)
        st.session_state["mcp_urls"] = conf.pipeline.mcp_urls
        st.success(f"MCP URL を {len(st.session_state['mcp_urls'])} 件読み込みました。")
    except Exception as e:
        st.error(f"設定読み込みエラー: {e}")
        st.session_state["mcp_urls"] = []

initial_prompt = st.text_area("初期プロンプト", height=200, value="頭が痛いです。何の病気の可能性がありますか？")
max_turns = st.slider("最大ターン数", min_value=1, max_value=10, value=5)

if st.button("議論開始"):
    urls = st.session_state.get("mcp_urls", [])
    if not urls:
        st.warning("MCP URL が設定されていません。")
    elif not initial_prompt.strip():
        st.warning("初期プロンプトを入力してください。")
    else:
        with st.spinner("議論中... 少々お待ちください"):
            history, summary = asyncio.run(
                run_discussion(initial_prompt, urls, max_turns, client)
            )

        for turn in history:
            with st.expander(f"Turn {turn['turn']}"):
                st.write("**Prompt:**", turn["prompt"])
                for res in turn["results"]:
                    st.write(f"- **{res['url']}**: {res['response']}")
        st.markdown("---")
        st.subheader("最終報告書")
        st.write(summary)
