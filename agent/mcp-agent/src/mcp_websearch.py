"""
Tavily Search MCP Server using SSE transport
"""
from mcp.server.fastmcp import FastMCP
import os
import httpx
import json
from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Server created
mcp = FastMCP(name="Tavily Search Tool")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API for current and relevant information.

    This tool should be used when:
    - Information beyond the knowledge cutoff date is needed
    - Current facts, news, or data are required
    - Real-time information is necessary
    - Verification of claims or statements is needed

    Usage guidelines:
    - Formulate clear, specific search queries for best results
    - For time-sensitive queries, include terms like 'today', 'this week', or 'recent'
    - Prefer concise queries (3-7 words) for more targeted results

    Args:
        query (str): A specific search query. Examples:
            - "latest AI research trends"
            - "current weather in New York"
            - "recent space exploration news"
        max_results (int): Maximum number of results to return (default: 5)

    Returns:
        str: Formatted search results with title, snippet, and URL
    """
    logger.info(f"Searching for: {query}")

    if not os.getenv("TAVILY_API_KEY",None):
        error_msg = "Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
        logger.error(error_msg)
        return error_msg

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.getenv("TAVILY_API_KEY",None),
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "include_images": False,
                    "search_depth": "basic"
                },
                timeout=30.0
            )

            data = response.json()

            # Format results
            results = []

            # Include the answer if available
            if data.get("answer"):
                results.append(f"**Answer**: {data['answer']}")
                results.append("")  # Empty line

            # Format search results
            for i, result in enumerate(data.get("results", []), 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "")
                url = result.get("url", "")

                results.append(f"{i}. **{title}**")
                results.append(f"   {snippet}")
                results.append(f"   URL: {url}")
                results.append("")  # Empty line between results

            formatted_results = "\n".join(results) if results else "No results found."
            logger.info(f"Found {len(data.get('results', []))} results")
            return formatted_results

    except httpx.HTTPError as e:
        error_msg = f"Search request failed: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error performing search: {str(e)}"
        logger.error(error_msg)
        return error_msg

if __name__ == "__main__":
    logger.info("Starting Tavily Search MCP server")
    port = int(os.getenv("SEARCH_PORT", "3002"))
    mcp.settings.port = port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")