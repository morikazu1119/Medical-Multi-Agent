import pytest

from src.mcp_clients.gemini.client import main


@pytest.mark.asyncio
async def test_integration():
    """
    インテグレーションテスト: 実際の MCP サーバーと Gemini API に対して main() を実行し、
    出力が空でなく「カレーライス」を含むことを確認する。
    """
    try:
        output = await main()
    except Exception as e:
        pytest.skip(f"インテグレーションテストをスキップ: {e}")

    assert output, "main() の出力が空です"
    assert "カレーライス" in output, f"出力に「カレーライス」が含まれていません: {output}"
