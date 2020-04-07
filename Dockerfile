FROM ubuntu:latest
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && apt-get update \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get update \
    && apt-get install -y tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Add User with Non Root Access
RUN addgroup --system idxpusergrp &&\
    adduser --system --ingroup idxpusergrp --home /home/idxpuser --shell /sbin/nologin idxpuser

WORKDIR /home/idxpuser/idxp
COPY ./requirements.txt /home/idxpuser/idxp/requirements.txt
RUN pip install -r requirements.txt
COPY . /home/idxpuser/idxp
RUN chown -R idxpuser:idxpusergrp /home/idxpuser/idxp
USER idxpuser
EXPOSE 9082
CMD ["python3", "-u", "api.py"]