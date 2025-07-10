import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

BASE_URL = "http://localhost:1111/mcp"  # テスト用のMCPサーバーURL


@pytest.mark.asyncio
async def test_recommend_dish_http():
    """正常系: シンプルな呼び出し"""
    async with Client(BASE_URL, timeout=5) as client:
        result = await client.call_tool("recommend_dish", {"request": "ping"})
        assert result.content[0].text == "おすすめはカレーライスです。"


@pytest.mark.asyncio
async def test_concurrent_requests():
    """正常系: 並列呼び出しでも壊れないか"""
    async with Client(BASE_URL, timeout=5) as client:
        tasks = [
            client.call_tool("recommend_dish", {"request": f"ping{i}"})
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert all(
            res.content and isinstance(res.content[0].text, str) for res in results
        )


@pytest.mark.asyncio
async def test_missing_parameters():
    """異常系: パラメータ不足時にToolError"""
    async with Client(BASE_URL) as client:
        with pytest.raises(ToolError, match="Field required"):
            await client.call_tool("recommend_dish", {})  # requestキーがない
