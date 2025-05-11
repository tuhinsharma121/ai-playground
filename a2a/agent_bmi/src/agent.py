from collections.abc import AsyncIterable
from typing import Any, Literal

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from pylogger import get_python_logger

logger = get_python_logger()

memory = MemorySaver()


def train_bmi_ml_model():
    """
    Train a BMI machine learning model using linear regression.

    This function reads BMI data from a CSV file, processes the data to
    separate features and target variables, and trains a linear regression
    model to predict the BMI index.

    The dataset is expected to contain columns: 'Height', 'Weight',
    and 'Index', where 'Index' is the target variable.

    The trained model can be used to predict BMI index based on the provided
    features.
    """

    # Data is downloaded from https://www.kaggle.com/datasets/rukenmissonnier/age-weight-height-bmi-analysis
    df = pd.read_csv("agent_bmi/src/bmi.csv")
    df.head()

    Y = df['BMI']
    X = df.drop(columns='BMI', axis=1)
    X.head(), Y.head()

    model = LinearRegression()
    model.fit(X, Y)
    return model


# Train the model
logger.info("Training BMI ML model")
model = train_bmi_ml_model()
logger.info("BMI ML model trained")
# Create FastAPI app


def invoke_bmi_agent(height: str, weight: str):
    """
    Invoke the BMI Agent tool.

    This function predicts the BMI index based on the provided height and weight.

    Args:
        height (float): The height in centimeters.
        weight (float): The weight in kilograms.

    Returns:
        str: A string containing the predicted BMI index.
    """
    logger.info("Invoking BMI Agent tool")

    try:
        height = float(height)
        weight = float(weight)
        X_test = pd.DataFrame({"height": [height], "weight": [weight]})
        prediction = model.predict(X_test)
        return str(prediction[0])

    except Exception as e:
        logger.error(f"Error invoking BMI Agent tool. Logs: {e}")
        return f"Error invoking BMI Agent tool. Logs: {str(e)}"


@tool
def bmi_tool(height: str,weight: str) -> str:
    """
    Calculates Body Mass Index (BMI) based on height and weight inputs.

    This tool provides a BMI calculation along with a basic health category classification
    following World Health Organization standards. BMI is a simple screening metric that
    measures body fat based on height and weight, applicable to adult men and women.

    Health interpretation ranges:
    - Below 18.5: Underweight
    - 18.5 to 24.9: Normal weight
    - 25.0 to 29.9: Overweight
    - 30.0 and above: Obesity

    Limitations:
    - BMI does not directly measure body fat or account for muscle mass
    - Not applicable for children, pregnant women, very athletic individuals, or the elderly
    - Should not be used as the sole diagnostic tool for health assessment

    Input requirements:
    - Height must be provided in centimeters (cm) as a numeric string
    - Weight must be provided in kilograms (kg) as a numeric string
    - Both inputs should contain only numbers (and optionally a decimal point)

    Args:
        height (str): Height in centimeters (e.g., "175" for 175 cm)
        weight (str): Weight in kilograms (e.g., "70.5" for 70.5 kg)

    Returns:
        str: A text string containing the calculated BMI value
    """

    logger.info("Invoking BMI Agent tool")
    response = invoke_bmi_agent(height, weight)
    logger.info(f"Response: {response}")
    return response


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class BMIAgent:
    SYSTEM_INSTRUCTION = """You are a specialized BMI (Body Mass Index) calculator assistant. Your primary purpose is to use the bmi_tool to calculate BMI based on user-provided height and weight. When users request BMI calculations, you should use bmi_tool with the height in centimeters and weight in kilograms, then provide comprehensive information about their BMI results. Follow these guidelines:
    - Always use 'bmi_tool' to calculate BMI before responding
    - Convert measurements to centimeters and kilograms if provided in other units
    - Provide the calculated BMI along with standard health category classifications
    - Include health implications and recommendations based on the BMI result
    - If user doesn't provide height or weight, request the missing information
    - Explain BMI limitations (doesn't account for muscle mass, age, gender, etc.)
    - Be encouraging and supportive regardless of BMI result
    Response status:
    - Set to "input_required" if height or weight is missing or if you need clarification
    - Set to "error" if there are technical issues with bmi_tool or invalid inputs
    - Set to "completed" when you have successfully calculated and provided the BMI results"""

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.tools = [bmi_tool]

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
                    'content': 'Performing BMI calculations...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the BMI results...',
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
