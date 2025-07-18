import base64
from io import BytesIO

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from PIL import Image

BASE_URL = "http://localhost:10002/mcp"


def make_test_image_b64(size=(100, 100), color=(255, 0, 0)) -> str:
    """指定サイズ・色の PNG を生成し、base64 エンコードして返す"""
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.mark.asyncio
async def test_invalid_base64_image_integration():
    """不正な base64 文字列を送ると ToolError（画像デコード失敗）が返る"""
    async with Client(BASE_URL, timeout=60) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "diagnose",
                {"request": {"image": "not-a-valid-base64!!!", "symptom": "頭痛があります"}},
            )
        assert "画像デコード失敗" in str(excinfo.value)


@pytest.mark.asyncio
async def test_text_only_integration():
    """テキストのみ送信した場合、診断テキストが返ってくる"""
    async with Client(BASE_URL, timeout=60) as client:
        result = await client.call_tool(
            "diagnose", {"request": {"symptom": "頭痛と発熱があります"}}
        )
    text = result.content[0].text
    assert isinstance(text, str)
    assert len(text) > 10


@pytest.mark.asyncio
async def test_image_and_text_integration():
    """画像＋テキストを送信した場合、診断テキストが返ってくる"""
    img_b64 = make_test_image_b64()
    async with Client(BASE_URL, timeout=60) as client:
        result = await client.call_tool(
            "diagnose", {"request": {"image": img_b64, "symptom": "傷口が赤く腫れています"}}
        )
    text = result.content[0].text
    assert isinstance(text, str)
    assert len(text) > 10
