from collections.abc import AsyncIterable
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Any, Literal
import resend
import os

from pylogger import get_python_logger

logger = get_python_logger()

memory = MemorySaver()


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


@tool
def email_tool(email_id: str, subject: str, body: str) -> str:
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


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class EmailAgent:
    SYSTEM_INSTRUCTION = \
        """
        You are a specialized email assistant that helps users format and send emails professionally. Your role includes:
        1. First, format raw content into well-structured HTML email bodies
        2. Then use 'email_tool' to send the formatted email
    
        Formatting guidelines:
        - Create clean, visually readable HTML for both desktop and mobile devices
        - Use appropriate HTML tags such as <h1>, <p>, <ul>, <li>, and <strong>
        - Preserve any important formatting (like lists or bullet points)
        - Do not change the words - only improve the HTML formatting
        - If user doesn't provide raw content, request it before proceeding
    
        Sending guidelines:
        - Use 'email_tool' only to send the already-formatted email
        - Ensure all required fields (recipient, subject, body etc.) are available before sending
        - Ask for missing information if needed
    
        Response status:
        - Set to "input_required" if email content or recipient details are missing
        - Set to "error" if there are technical issues with email_tool or invalid inputs
        - Set to "completed" when you have successfully formatted and sent the email
        """

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.tools = [email_tool]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def invoke(self, query, sessionId) -> str:
        config = {'configurable': {'thread_id': sessionId}}
        self.graph.invoke({'messages': [('user', query)]}, config)
        return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Sending Email...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Sending Email...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
                structured_response, ResponseFormat
        ):
            if (
                    structured_response.status == 'input_required'
                    or structured_response.status == 'error'
            ):
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
