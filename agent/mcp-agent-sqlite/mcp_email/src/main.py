# Import dependencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP(name="Email Agent Tool")

# Import all the tools
import resend
import os

from utils.pylogger import get_python_logger

logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))


def invoke_email_agent(email_id: str, subject: str, body: str):
    """
    This function sends an email using Resend.

    Args:
        email_id (str): The email address to send the email to.
        subject (str): The subject of the email.
        body (str): The body of the email.

    Returns:
        str: A dictionary containing the status of the email sent.
    """
    try:

        # Send email using Resend
        resend.api_key = os.environ.get("RESEND_API_KEY")

        # Send the email
        params = {
            "from": "Acme <onboarding@resend.dev>",
            "to": [os.getenv("RESEND_EMAIL_ID", "tuhinsharma121@gmail.com")],
            "subject": subject,
            "html": body,
        }
        logger.info(f"Sending email with params: {params} to {email_id}")
        resend.Emails.send(params)
        logger.info("Email sent successfully.")
        return "Email sent successfully."
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return f"Error sending email: {e}"


@mcp.tool()
async def email_tool(email_id: str, subject: str, body: str) -> str:
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
        email_id (str): The email address to send the email to
        subject (str): The subject line of the email (required, max 255 characters)
                       Should be descriptive and relevant to the content

        body (str): The main content of the email (required, max 1000 characters)
                    Should include a proper greeting and signature

    Returns:
        str: A text string containing the email delivery status and confirmation
    """

    logger.info("Invoking Email Agent tool")
    logger.info(f"Email ID: {email_id}, Subject: {subject}, Body: {body}")
    response = invoke_email_agent(email_id=email_id, subject=subject, body=body)
    logger.info(f"Response: {response}")
    logger.info("Email Agent tool invoked")
    return response


if __name__ == "__main__":
    logger.info("Starting Email Agent MCP server")
    port = int(os.getenv("PORT", "2002"))
    mcp.settings.port = port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")
