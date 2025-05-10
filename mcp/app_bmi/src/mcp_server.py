# Import dependencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP(name="BMI Agent Tool")

# Import all the tools
from app_bmi.src.tools import *

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    logger.info("Starting BMI Agent MCP server")
    mcp.settings.port = int(os.getenv("PORT", "1002"))
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")
