import json
import os

import requests

from app_email.src.mcp_server import mcp
from pylogger import get_python_logger

logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))


@mcp.tool()
def email_agent_tool(subject: str, body: str) -> str:
    """
    Sends emails through the organization's email system to recipients.

    This tool allows composing and sending emails with customized subject lines and content.
    The system automatically adds the sender's information and properly formats the email.

    Usage guidelines:
    - Keep subject lines clear and concise
    - Ensure body content is professional and complete

    Limitations:
    - Cannot send attachments through this interface
    - Emails are sent from the system's default account

    Args:
        subject (str): The subject line of the email (required, max 255 characters)
                       Should be descriptive and relevant to the content

        body (str): The main content of the email (required, max 1000 characters)
                    Should include a proper greeting and signature

    Returns:
        str: A text string containing the email delivery status and confirmation
    """

    logger.info("Invoking Email Agent tool")
    payload = {"subject": subject, "body": body}
    logger.info(f"Payload: {payload}")
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "3001"))
    response = requests.post(
        url=f"http://{host}:{port}/invoke",
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    logger.info(f"Response: {response.json()}")
    return response.json()
