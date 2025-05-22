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
COPY agent_redhat/pyproject.toml /app/pyproject.toml
RUN pip install uv
RUN uv venv
RUN source .venv/bin/activate
RUN uv pip install -r /app/pyproject.toml

# --------------------------------------------------------------------------------------------------
# copy src code and files into dir /root
# --------------------------------------------------------------------------------------------------

COPY agent_redhat /app/agent_redhat
COPY utils /app/utils

# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

ADD agent_redhat/deployment/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]