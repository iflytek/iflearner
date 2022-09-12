FROM python:3.9

ARG IFLEARNER_DIR=/iflearner
COPY requirements.txt $IFLEARNER_DIR/
COPY iflearner $IFLEARNER_DIR/iflearner
COPY examples $IFLEARNER_DIR/examples
ENV PYTHONPATH=$IFLEARNER_DIR
WORKDIR $IFLEARNER_DIR
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
