# Import dependencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP(name="Email Agent Tool")

# Import all the tools
from app_email.src.tools import *

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    logger.info("Starting Email Agent MCP server")
    port = int(os.getenv("PORT", "3002"))
    mcp.settings.port = port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")
