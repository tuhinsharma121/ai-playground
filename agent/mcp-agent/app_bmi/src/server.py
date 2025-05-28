import os

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

from pylogger import get_python_logger,get_uvicorn_log_config

# Set up logging
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
    df = pd.read_csv("app_bmi/src/bmi.csv")
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
app = FastAPI(title="BMI Agent API", description="API for a BMI Agent")


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


class Query(BaseModel):
    """Query"""
    height: str
    weight: str


class Response(BaseModel):
    """Response"""
    response: str


@app.get("/")
def read_root():
    """Read root"""
    logger.info("Read root")
    return {"Hey": "BMI Agent"}


@app.post("/invoke")
def get_bmi_agent_response(query: Query):
    """Get BMI Agent response"""
    logger.info("Invoking BMI Agent tool")
    logger.info(f"Query: {query}")
    response = invoke_bmi_agent(height=query.height, weight=query.weight)
    response = Response(response=response)
    logger.info(f"Response: {response}")
    return response


if __name__ == "__main__":
    logger.info("Starting BMI Agent server")
    port = int(os.getenv("PORT", "1001"))
    uvicorn.run(app, host="0.0.0.0", port=port,log_config=get_uvicorn_log_config())
