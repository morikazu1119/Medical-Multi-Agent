import argparse

import uvicorn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sample-mcp")


@mcp.tool()
def recommend_dish(request):
    return "おすすめはカレーライスです。"


def main():
    parser = argparse.ArgumentParser(description="Sample FastMCP Server")
    parser.add_argument(
        "--host", type=str, required=True, help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port to bind the server"
    )
    args = parser.parse_args()

    uvicorn.run(mcp.streamable_http_app, host=args.host, port=int(args.port), workers=1)


if __name__ == "__main__":
    main()
