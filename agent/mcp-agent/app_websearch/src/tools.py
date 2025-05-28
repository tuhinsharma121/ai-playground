import json
import os

import requests

from pylogger import get_python_logger
from app_websearch.src.mcp_server import mcp

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))


@mcp.tool()
def websearch_agent_tool(query: str) -> str:
    """
    Searches the web for current and relevant information based on user queries.

    This tool should be used when:
    - Information beyond the knowledge cutoff date is needed
    - Current facts, news, or data are required
    - Verification of claims or statements is necessary

    Usage guidelines:
    - Formulate clear, specific search queries for best results
    - For time-sensitive queries, include terms like 'today', 'this week', or 'recent'
    - Prefer concise queries (3-7 words) for more targeted results

    Args:
        query (str): A specific search query. Examples:
            - "latest AI research trends"
            - "current weather in New York"
            - "recent space exploration news"

    Returns:
        str: A text string containing the search results. The string may include:
            - Snippets from relevant websites
            - Brief summaries of found information
            - Source URLs when available
    """

    logger.info("Invoking WebSearch Agent tool")
    payload = {"query": query}
    logger.info(f"Payload: {payload}")
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "5001"))
    response = requests.post(
        url=f"http://{host}:{port}/invoke",
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    logger.info(f"Response: {response.json()}")
    return response.json()
