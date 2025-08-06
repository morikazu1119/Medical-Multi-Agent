import asyncio
import logging
import re

from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.1
SUMMARY_PROMPT = "以下は複数のMCPサーバーからの応答です。日本語で要約してください。"
JUDGE_PROMPT = "あなたはモデレーターです。以下を分析し、'CONTINUE' か 'STOP' を返してください。"


async def call_mcp(url: str, prompt: str, client: genai.Client) -> str:
    """
    Gemini を使用して MCP サーバーにリクエストを送信し、応答を受け取る。
    Args:
        url (str): MCP サーバーの URL。
        prompt (str): リクエストの内容。
        client (genai.Client): Gemini クライアント。
    Returns:
        str: MCP サーバーからの応答。
    """
    logger.info(f"Using model is {GEMINI_MODEL}")
    logger.info(f"Calling MCP: {url}")
    try:
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        tools=[session],
                    ),
                )
                logger.info("Calling MCP completed.")
                return response.text
    except Exception as e:
        err_msg = f"{url}: {e}"
        logger.error(err_msg)
        return err_msg


async def summarize(responses: list[str], client: genai.Client) -> str:
    """
    MCPの応答を結合し、Gemini API で日本語の要約を生成して出力。
    Args:
        responses (list[str]): MCP サーバーからの応答のリスト。
        client (genai.Client): Gemini クライアント。
    Returns:
        str: 要約されたテキスト。
    """
    combined_text = "\n\n".join(responses)
    combined_summary_prompt = SUMMARY_PROMPT + combined_text
    logger.info("Generating summary from responses.")
    summary = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=combined_summary_prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    logger.info("Summary generation completed.")
    return summary.text


async def summarize_history(history: list[dict], client: genai.Client) -> str:
    combined = "\n\n".join(
        [
            f"--- ターン {t['turn']} ---\nPrompt: {t['prompt']}\n"
            + "\n".join(
                f"Response from {r['url']}: {r['response']}" for r in t["results"]
            )
            for t in history
        ]
    )
    prompt = "議論履歴を統合し、診断仮説・検査計画・治療方針を構造化して報告してください。" + combined
    out = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    return out.text


async def judge_continue(history: list[dict], client: genai.Client) -> bool:
    """
    ジャッジプロンプトを使用して、ディスカッションを続けるかどうかを判断。
    Args:
        history (list[dict]): ディスカッションの履歴。
        client (genai.Client): Gemini クライアント。
    Returns:
        bool: ディスカッションを続ける場合は True、停止する場合は False。
    """
    logger.info("Judging whether to continue the discussion.")
    summary = "\n\n".join(
        [
            f"--- ターン {t['turn']} ---\nPrompt: {t['prompt']}\n"
            + "\n".join(
                f"Response from {r['url']}: {r['response']}" for r in t["results"]
            )
            for t in history
        ]
    )
    resp = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=JUDGE_PROMPT + summary,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    logger.info("Judgment completed.")
    m = re.search(r"(CONTINUE|STOP)", resp.text, re.IGNORECASE)
    return m and m.group(1).upper() == "CONTINUE"


async def run_discussion(
    initial: str, urls: list[str], max_turns: int, client: genai.Client
):
    """
    複数のMCPサーバーとディスカッションを行い、要約を生成する。
    Args:
        initial (str): 初期プロンプト。
        urls (list[str]): MCPサーバーのURLリスト。
        max_turns (int): 最大ターン数。
        client (genai.Client): Gemini クライアント。
    Returns:
        tuple: ディスカッションの履歴と要約。
    """
    logger.info("Starting discussion with MCP servers.")
    history = []
    current = initial
    for i in range(max_turns):
        turn = i + 1
        responses = await asyncio.gather(*[call_mcp(u, current, client) for u in urls])
        results = [{"url": u, "response": r} for u, r in zip(urls, responses)]
        history.append({"turn": turn, "prompt": current, "results": results})

        if not await judge_continue(history, client):
            break

        merged = "\n\n".join([f"{r['url']}: {r['response']}" for r in results])
        current = f"ターン{turn+1}のプロンプト\n{merged}"
    summary = await summarize_history(history, client)
    logger.info("Discussion completed.")
    return history, summary
