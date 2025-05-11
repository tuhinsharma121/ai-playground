FROM docker.io/library/python:3.13.2
LABEL maintainer="Tuhin Sharma <tuhinsharma121@gmail.com>"

# --------------------------------------------------------------------------------------------------
# set the working directory to /
# --------------------------------------------------------------------------------------------------

WORKDIR /

# --------------------------------------------------------------------------------------------------
# Copy manifest files into dir /
# install python packages
# --------------------------------------------------------------------------------------------------

COPY agent_websearch/deployment/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# --------------------------------------------------------------------------------------------------
# copy src code and files into dir /root
# --------------------------------------------------------------------------------------------------

COPY agent_websearch /agent_websearch
COPY agent_websearch/src/main.py /main.py
COPY common /common
COPY pylogger.py /pylogger.py

# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

ADD agent_websearch/deployment/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]