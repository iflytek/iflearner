# Jupyterlab

JupyterLab是Jupyter主打的最新数据科学生产工具，某种意义上，它的出现是为了取代Jupyter Notebook。不过不用担心Jupyter Notebook会消失，
JupyterLab包含了Jupyter Notebook所有功能。

JupyterLab作为一种基于web的集成开发环境，你可以使用它编写notebook、操作终端、编辑markdown文本、打开交互模式、查看csv文件及图片等功能。

## 示例
该实例主要演示如何在jupyterlab中安装和运行iflearner库，并且启动一个server端和两个client端，完成在mnist数据集下的图像分类的联邦学习任务。

安装启动顺序为: 1. Server端 2. Client端。

### 1. Server端
请参阅`pytorch_mnist_server.ipynb`，进行server的安装和启动

### 2. Client端
1. 请参阅`pytorch_mnist_client1.ipynb`，进行client1的安装和启动
2. 请参阅`pytorch_mnist_client2.ipynb`，进行client2的安装和启动

当两个client都启动注册后，将自动开启联邦训练任务。