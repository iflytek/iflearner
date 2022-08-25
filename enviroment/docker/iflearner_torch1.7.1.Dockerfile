FROM python:3.9

RUN pip3 install --upgrade pip && groupadd -r iflearner && useradd -r -g iflearner iflearner && \
pip3 install iflearner tensorflow==2.9.1 torchvision==0.8.2 --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

USER iflearner