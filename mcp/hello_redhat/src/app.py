import base64
import json
import os

import requests
import streamlit as st

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Load and encode the image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


img_base64 = get_base64_image("agent_cowboy/src/cowboy.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{img_base64}" width="200"/>
        <h1>Hey Cowboy</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Create a form
with st.form(key="query_form"):
    # User input inside the form
    query = st.text_input("")

    # Submit button
    submit_button = st.form_submit_button(label="Ask")

# Process the query only when the form is submitted
if submit_button and query:
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8000"))
    payload = {"query": query}
    logger.info(f"Payload: {payload}")
    response = requests.post(
        url=f"http://{host}:{port}/query",
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    logger.info(f"Response: {response}")
    # Get the answer using the question answering pipeline
    res = {"answer": response.json()["response"]}
    logger.info(f"Answer: {res}")
    # Display the answer
    st.text_area("Fedora's resolution:", res['answer'])
