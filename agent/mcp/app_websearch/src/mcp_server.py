# Import depdendencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP(name="Websearch Agent Tool")

# Import all the tools
from app_websearch.src.tools import *

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    logger.info("Starting WebSearch Agent MCP server")
    port = int(os.getenv("PORT", "5002"))
    mcp.settings.port = port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")
