import base64
from io import BytesIO

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from PIL import Image

# Docker コンテナで起動した MCP サーバーのエンドポイント
BASE_URL = "http://localhost:10004/mcp"


def make_test_image_b64() -> str:
    """100×100 の赤い PNG を生成し、base64 エンコードして返す"""
    buf = BytesIO()
    Image.new("RGB", (100, 100), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.mark.asyncio
async def test_invalid_base64_image_integration():
    """不正な base64 文字列を送ると ToolError（画像デコード失敗）が返る"""
    async with Client(BASE_URL, timeout=10) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "diagnose",
                {"request": {"image": "not-a-valid-base64!!!", "symptom": "頭痛があります"}},
            )
        assert "失敗" in str(excinfo.value)


@pytest.mark.asyncio
async def test_text_only_integration():
    """テキストのみ送信した場合、診断テキストが返ってくる"""
    async with Client(BASE_URL, timeout=120) as client:
        result = await client.call_tool(
            "diagnose", {"request": {"symptom": "頭痛と発熱があります"}}
        )
    # content はリスト形式なので最初の TextContent の .text を取り出して検証
    text = result.content[0].text
    assert isinstance(text, str)
    assert len(text) > 10  # 十分な長さがあることの簡易チェック


@pytest.mark.asyncio
async def test_image_and_text_integration():
    """画像＋テキストを送信した場合、診断テキストが返ってくる"""
    img_b64 = make_test_image_b64()
    async with Client(BASE_URL, timeout=120) as client:
        result = await client.call_tool(
            "diagnose", {"request": {"image": img_b64, "symptom": "傷口が赤く腫れています"}}
        )
    text = result.content[0].text
    assert isinstance(text, str)
    assert len(text) > 10
