from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from pylogger import get_python_logger

logger = get_python_logger()

memory = MemorySaver()


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
        web_search_tool = TavilySearchResults(k=5)
        docs = web_search_tool.invoke({"query": query})
        result = [d["content"] for d in docs]
        return result

    except Exception as e:
        logger.error(f"An error occurred while while invoking TavilySearchResults. Logs: {str(e)}")
        return f"An error occurred while while invoking TavilySearchResults. Logs: {str(e)}"


@tool
def websearch_tool(query: str) -> str:
    """
    Searches the web for current and relevant information based on user queries.

    This tool should be used when:
    - Information beyond the knowledge cutoff date is needed
    - Current facts, news, or data are required
    - Verification of claims or statements is necessary

    Usage guidelines:
    - Formulate clear, specific search queries for best results
    - For time-sensitive queries, include terms like 'today', 'this week', or 'recent'
    - Prefer concise queries (3-7 words) for more targeted results

    Args:
        query (str): A specific search query. Examples:
            - "latest AI research trends"
            - "current weather in New York"
            - "recent space exploration news"

    Returns:
        docs: A list of text strings containing the search results.
    """

    logger.info("Invoking WebSearch Agent tool")
    logger.info(f"Query: {query}")
    response = invoke_websearch_agent(query)
    logger.info(f"Response: {response}")
    logger.info("WebSearch Agent tool invoked")
    return response


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class WebsearchAgent:
    SYSTEM_INSTRUCTION = """You are a specialized web search assistant. Your primary purpose is to use the websearch_tool to find and retrieve information from the internet. When users ask questions, you should use websearch_tool to search for relevant information and provide comprehensive answers based on your search results. Follow these guidelines:
    - Always use 'websearch_tool' to search for current and accurate information before responding
    - Provide comprehensive answers based on the search results
    - Cite your sources with proper links and attributions
    - If websearch_tool cannot find information, inform the user about this limitation
    - For topics that require specialized functionality beyond web search, explain your constraints
    - Be thorough but concise in your responses
    Response status:
    - Set to "input_required" if you need clarification or more specific details from the user
    - Set to "error" if there are technical issues with websearch_tool or invalid queries
    - Set to "completed" when you have successfully found and provided the requested information"""

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.tools = [websearch_tool]

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
                    'content': 'Performing web search...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the web search results...',
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
