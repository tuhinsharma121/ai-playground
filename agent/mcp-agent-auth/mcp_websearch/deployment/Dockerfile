FROM python:3.13.3
LABEL maintainer="Tuhin Sharma <tuhinsharma121@gmail.com>"

# --------------------------------------------------------------------------------------------------
# set the working directory to /
# --------------------------------------------------------------------------------------------------

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

# --------------------------------------------------------------------------------------------------
# Copy manifest files into dir /
# install python packages
# --------------------------------------------------------------------------------------------------

SHELL ["/bin/bash", "-c"]
COPY mcp_websearch/pyproject.toml /app/pyproject.toml
RUN pip install uv
RUN uv venv
RUN source /app/.venv/bin/activate
RUN uv pip install -r /app/pyproject.toml

# --------------------------------------------------------------------------------------------------
# copy src code and files into dir /root
# --------------------------------------------------------------------------------------------------

COPY mcp_websearch /app/mcp_websearch
COPY utils /app/utils

# --------------------------------------------------------------------------------------------------
# add healthcheck for the container
# --------------------------------------------------------------------------------------------------

ADD mcp_websearch/deployment/healthcheck.sh /bin/healthcheck.sh
RUN chmod +x /bin/healthcheck.sh

# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

ADD mcp_websearch/deployment/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]