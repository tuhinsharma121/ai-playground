FROM docker.io/library/python:3.11.12
LABEL maintainer="Tuhin Sharma <tuhsharm@redhat.com>"

# --------------------------------------------------------------------------------------------------
# configure Red Hat IT CA certs
# --------------------------------------------------------------------------------------------------

RUN pip3 install --disable-pip-version-check --no-cache-dir poetry==1.5.1
RUN curl --create-dirs -O --output-dir ~/.rh_certs https://certs.corp.redhat.com/certs/Current-IT-Root-CAs.pem
RUN poetry config certificates.de-cop-nexus.cert ~/.rh_certs/Current-IT-Root-CAs.pem

# --------------------------------------------------------------------------------------------------
# set the working directory to /
# --------------------------------------------------------------------------------------------------

WORKDIR /

# --------------------------------------------------------------------------------------------------
# Copy manifest files into dir /
# install python packages
# --------------------------------------------------------------------------------------------------

COPY agent_cowboy/deployment/agent/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# --------------------------------------------------------------------------------------------------
# copy src code and files into dir /root
# --------------------------------------------------------------------------------------------------

COPY agent_cowboy /agent_cowboy
COPY agent_cowboy/src/server.py /server.py
COPY pylogger.py /pylogger.py

# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

ADD agent_cowboy/deployment/agent/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]