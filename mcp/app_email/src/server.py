import os

import resend
import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from pylogger import get_python_logger, get_uvicorn_log_config

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Create FastAPI app
app = FastAPI(title="Email Agent API", description="API for an Email Agent")


def invoke_email_agent(subject: str, body: str):
    """
    This function sends an email using Resend.

    Args:
        subject (str): The subject of the email.
        body (str): The body of the email.

    Returns:
        str: A dictionary containing the status of the email sent.
    """
    try:
        logger.info(f"subject: {subject}")
        logger.info(f"body: {body}")

        # Create a PromptTemplate
        prompt = PromptTemplate.from_template("""
            You are an expert email copywriter and HTML formatter.
    
            Your task is to take the content below and generate a professional, well-formatted HTML email body. 
            The email should be clean, visually readable on desktop and mobile, and use basic HTML only (no CSS or JavaScript). 
            
            - Use appropriate HTML tags such as <h1>, <p>, <ul>, <li>, and <strong>.
            - Preserve any important formatting (like lists or bullet points).
            - Do not change the words. Only change the HTML formatting.
            
            Here is the raw content you need to format:
            
            {content}
            """)

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build a Runnable pipeline
        chain: Runnable = prompt | llm

        # Invoke the chain
        result = chain.invoke({"content": body})
        html_body = result.content
        logger.info(f"formatted HTML body: {html_body}")

        # Send email using Resend
        resend.api_key = os.environ.get("RESEND_API_KEY")

        # Send the email
        params = {
            "from": "Acme <onboarding@resend.dev>",
            "to": [os.getenv("RESEND_EMAIL_ID", "tuhinsharma121@gmail.com")],
            "subject": subject,
            "html": html_body,
        }
        logger.info(f"Sending email with params: {params}")
        resend.Emails.send(params)
        logger.info("Email sent successfully.")
        return "Email sent successfully."
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return f"Error sending email: {e}"


class Query(BaseModel):
    subject: str
    body: str


class Response(BaseModel):
    response: str


@app.get("/")
def read_root():
    logger.info("Read root")
    return {"Hey": "Email Agent"}


@app.post("/invoke")
def get_email_agent_response(query: Query):
    """Get Email Agent response"""
    logger.info("Invoking Email Agent tool")
    logger.info(f"Query: {query}")
    response = invoke_email_agent(subject=query.subject, body=query.body)
    response = Response(response=response)
    logger.info(f"Response: {response}")
    return response


if __name__ == "__main__":
    logger.info("Starting Email Agent server")
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=get_uvicorn_log_config())
