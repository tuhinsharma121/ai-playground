import os

import uvicorn
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from pylogger import get_python_logger, get_uvicorn_log_config

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))
app = FastAPI(title="WebSearch Agent API", description="API for a WebSearch Agent")


def invoke_websearch_agent(query):
    """
    This function uses OpenAI's GPT-3.5-turbo model to generate a response to the provided query.

    Args:
        query (str): The query to be answered by the WebSearch Agent.

    Returns:
        str: The generated response from the WebSearch Agent.

    """

    try:
        logger.info("Invoking WebSearch Agent tool")
        web_search_tool = TavilySearchResults(k=3)
        docs = web_search_tool.invoke({"query": query})
        context = "\n".join([d["content"] for d in docs])

        template = """
        You are a helpful AI assistant. Use the following context to answer the user's question.

        Context:
        {context}

        Question:
        {question}

        Answer in a concise and informative way.
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        # Example usage
        final_prompt = prompt.format(
            context=context,
            question=query
        )

        llm = OpenAI(model="gpt-4o-mini")
        response = llm.complete(final_prompt)
        logger.info(f"Response: {response.text}")
        return response.text

    except Exception as e:
        logger.error(f"An error occurred while while invoking TavilySearchResults. Logs: {str(e)}")
        return f"An error occurred while while invoking TavilySearchResults. Logs: {str(e)}"


class Query(BaseModel):
    """Query model"""
    query: str


class Response(BaseModel):
    """Response model"""
    response: str


@app.get("/")
def read_root():
    """Read root"""
    logger.info("Read root")
    return {"Hello": "Websearch Agent"}


@app.post("/invoke")
def get_websearch_agent_response(query: Query):
    """Get WebSearch Agent response"""
    logger.info("Invoking WebSearch Agent tool")
    logger.info(f"Query: {query}")
    response = invoke_websearch_agent(query=query.query)
    response = Response(response=response)
    logger.info(f"Response: {response}")
    return response


if __name__ == "__main__":
    logger.info("Starting WebSearch Agent server")
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=get_uvicorn_log_config())
