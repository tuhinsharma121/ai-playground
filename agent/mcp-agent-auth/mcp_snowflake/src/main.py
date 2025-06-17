#!/usr/bin/env python
import json
import os
import time
from typing import Any

import snowflake.connector
from dotenv import load_dotenv
from fastmcp import Context, FastMCP

from utils.pylogger import get_python_logger

logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

load_dotenv()


def execute_query(sf_access_token: str, query: str) -> list[dict[str, Any]]:
    """
    Execute SQL query and return results

    Args:
        sf_access_token (str) : Snowflake access token
        query (str): SQL query statement

    Returns:
        list[dict[str, Any]]: List of query results
    """
    start_time = time.time()

    config = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "token": sf_access_token,  # Your OIDC access token
        "authenticator": "oauth",  # Changed from "externalbrowser" to "oauth"
    }
    logger.info(
        f"Executing query: {query[:200]}..."
    )  # 只记录前200个字符 / Log only first 200 characters

    try:
        conn = snowflake.connector.connect(
            **config,  # Fixed: should be config, not self.config
            client_session_keep_alive=True,
            network_timeout=30,
            login_timeout=30,
        )
        with conn.cursor() as cursor:
            # Use transaction for write operations
            if any(
                query.strip().upper().startswith(word)
                for word in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
            ):
                cursor.execute("BEGIN")
                try:
                    cursor.execute(query)
                    conn.commit()
                    logger.info(
                        f"Write query executed in {time.time() - start_time:.2f}s"
                    )
                    return [{"affected_rows": cursor.rowcount}]
                except Exception:
                    conn.rollback()
                    raise
            else:
                # Read operations
                cursor.execute(query)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    results = [dict(zip(columns, row)) for row in rows]
                    logger.info(
                        f"Read query returned {len(results)} rows in {time.time() - start_time:.2f}s"
                    )
                    return results
                return []

    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"SQL Error: {str(e)}")
        logger.error(f"Error Code: {getattr(e, 'errno', 'unknown')}")
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


mcp = FastMCP("Snowflake MCP Server")


def get_bearer_token(ctx):
    request = ctx.get_http_request()
    headers = request.headers
    # Check if 'Authorization' header is present
    authorization_header = headers.get("Authorization")

    if authorization_header:
        # Split the header into 'Bearer <token>'
        parts = authorization_header.split()

        if len(parts) == 2 and parts[0] == "Bearer":
            return parts[1]
        else:
            raise ValueError("Invalid Authorization header format")
    else:
        raise ValueError("Authorization header missing")


@mcp.tool()
def tool_snowflake(query: str, ctx: Context) -> str:
    """
    Execute a SQL query on Snowflake

    Args:
        sf_access_token (str) : Snowflake access token
    logger.info(f"access_token: {access_token}")

        query: SQL query to execute

    Returns:
        Query results as formatted string
    """
    start_time = time.time()
    token = get_bearer_token(ctx)
    logger.info(f"Token: {token}")

    try:
        result = execute_query(token, query)
        execution_time = time.time() - start_time

        return f"Results (execution time: {execution_time:.2f}s):\n{json.dumps(result, indent=2, default=str)}"
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        logger.error(error_message)
        return error_message


def cleanup():
    """
    Clean up resources, close database connection
    """
    db.close()


if __name__ == "__main__":
    import atexit

    # Register cleanup function
    atexit.register(cleanup)

    try:
        # Configure server options
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "4002"))

        logger.info(f"Starting FastMCP server on {host}:{port}")

        # Start server
        mcp.run(transport="streamable-http", host=host, port=port)

    except Exception as e:
        logger.critical(f"Server failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Server shutting down")
        cleanup()

    cursor = conn.cursor()

    query = """SELECT * FROM SNOWFLAKE_LEARNING_DB.TUHIN_MART.EMPLOYEES"""
    cursor.execute(query)
    conn.commit()
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, row)) for row in rows]
    logger.info(results)
