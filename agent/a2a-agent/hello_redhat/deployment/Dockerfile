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

COPY hello_redhat/deployment/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# --------------------------------------------------------------------------------------------------
# copy src code and files into dir /root
# --------------------------------------------------------------------------------------------------

COPY hello_redhat /hello_redhat
COPY hello_redhat/src/app.py /app.py
COPY common /common
COPY pylogger.py /pylogger.py

# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

ADD hello_redhat/deployment/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]