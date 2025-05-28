import asyncio
import argparse
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client  # Streamable HTTP client


async def run_client(server_url):
    try:
        async with streamablehttp_client(server_url) as (
                client_read_stream,
                client_write_stream,
                client_get_session_id_callback
        ):
            async with ClientSession(client_read_stream, client_write_stream) as session:
                await session.initialize()  # Initialize the session
                await session.send_ping()

    except Exception as e:
        print(f"Error connecting to {server_url}:", e)


def main():
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument(
        "--server-url",
        "-u",
        help="MCP server URL"
    )

    args = parser.parse_args()
    asyncio.run(run_client(args.server_url))


if __name__ == "__main__":
    main()