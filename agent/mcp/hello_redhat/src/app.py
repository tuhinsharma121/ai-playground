import base64
import json
import os
import sys

import requests
import streamlit as st

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Set Streamlit configuration at the top of the file
# This must be done before any other st commands
st.set_page_config(page_title="Hello Red Hat", page_icon="🎩")


# Load and encode the image in base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        logger.error(f"Image not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


# Try to load the image with error handling
img_path = "hello_redhat/src/fedora.png"
if not os.path.exists(img_path):
    # Try alternative paths
    alt_paths = ["src/fedora.png", "fedora.png", "/app/hello_redhat/src/fedora.png"]
    for path in alt_paths:
        if os.path.exists(path):
            img_path = path
            break

img_base64 = get_base64_image(img_path)

# Display the header with or without the image
if img_base64:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" width="200"/>
            <h1>Hello Red Hat</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Hello Red Hat</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Create a form
with st.form(key="query_form"):
    # User input inside the form
    query = st.text_input("Enter your query:")

    # Submit button
    submit_button = st.form_submit_button(label="Ask")

# Process the query only when the form is submitted
if submit_button and query:
    try:
        host = os.getenv("AGENT_HOST", "0.0.0.0")
        port = int(os.getenv("AGENT_PORT", "8000"))
        payload = {"query": query}

        logger.info(f"Sending request to {host}:{port}")
        logger.info(f"Payload: {payload}")

        # Use timeout to prevent hanging
        response = requests.post(
            url=f"http://{host}:{port}/query",
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30  # 30 second timeout
        )

        response.raise_for_status()  # Raise exception for HTTP errors

        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response: {response.text}")

        # Get the answer from the response
        response_data = response.json()
        answer = response_data.get("response", "No response received")

        # Display the answer
        st.text_area("Fedora's resolution:", answer, height=150)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        st.error(f"Error connecting to the agent: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        st.error("Error parsing the response from the agent")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred: {e}")

# Add a footer with connection info (useful for debugging)
st.markdown("---")
st.markdown(f"**Agent endpoint:** `{os.getenv('AGENT_HOST', '0.0.0.0')}:{os.getenv('AGENT_PORT', '8000')}`")