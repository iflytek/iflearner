FROM captainji/jupyterlab:3.0.5

RUN pip3 install --upgrade pip && pip3 install iflearner \
--extra-index-url http://pypi.douban.com/simple \
--trusted-host pypi.douban.com