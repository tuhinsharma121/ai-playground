# Import dependencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP(name="BMI Agent Tool")

# Import all the tools
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

import requests

from pylogger import get_python_logger

logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

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
    df = pd.read_csv("src/bmi.csv")
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

@mcp.tool()
async def bmi_tool(height: str,weight: str) -> str:
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
    logger.info(f"Height: {height}, Weight: {weight}")
    response = invoke_bmi_agent(height, weight)
    logger.info(f"Response: {response}")
    logger.info("Invoking BMI Agent tool completed")
    return response


# Set up logging

if __name__ == "__main__":
    logger.info("Starting BMI Agent MCP server")
    mcp.settings.port = int(os.getenv("PORT", "1002"))
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="sse")
