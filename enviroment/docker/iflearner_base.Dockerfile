FROM python:3.9

RUN pip3 install --upgrade pip && groupadd -r iflearner && useradd -r -g iflearner iflearner && \
pip3 install iflearner --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

USER iflearner

