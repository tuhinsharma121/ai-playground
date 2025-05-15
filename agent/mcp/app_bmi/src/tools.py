import json
import os

import requests

from app_bmi.src.mcp_server import mcp

from pylogger import get_python_logger
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))


@mcp.tool()
def bmi_agent_tool(height: str,weight: str) -> str:
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
    payload = {"height": height, "weight": weight}
    logger.info(f"Payload: {payload}")
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "1001"))
    response = requests.post(
        url=f"http://{host}:{port}/invoke",
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    logger.info(f"Response: {response.json()}")
    return response.json()
