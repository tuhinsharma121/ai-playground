"""
Calculator MCP Server using SSE transport
"""
from mcp.server.fastmcp import FastMCP
import numexpr
import math
import re
import os
from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Server created
mcp = FastMCP(name="Calculator Tool")

@mcp.tool()
async def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions using numexpr.

    This tool should be used when:
    - Mathematical calculations are needed
    - Complex expressions need to be evaluated
    - Numeric operations are required

    Usage guidelines:
    - Use standard mathematical notation
    - Supports basic arithmetic operations (+, -, *, /, **)
    - Includes constants like pi and e
    - Supports functions like sqrt, sin, cos, tan, log

    Args:
        expression (str): A valid mathematical expression. Examples:
            - "2 + 2"
            - "sqrt(16)"
            - "pi * (5**2)"
            - "log(100)"

    Returns:
        str: The result of the mathematical calculation
    """
    logger.info(f"Calculating: {expression}")
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        result = re.sub(r"^\[|\]$", "", output)
        logger.info(f"Result: {result}")
        return result
    except Exception as e:
        error_msg = f'calculator("{expression}") raised error: {e}. Please try again with a valid numerical expression'
        logger.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    logger.info("Starting Calculator MCP server")
    port = int(os.getenv("CALCULATOR_PORT", "8080"))
    mcp.settings.port = port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")