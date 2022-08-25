# iFlearner Example using Mxnet

该示例将帮助您了解如何在 mxnet 下运行联邦任务

## 前提条件
首先需要安装iflearner，然后安装对应的依赖
```shell
pip install iflearner
```

安装相应的依赖项。
```shell
pip install -r requirements.txt
```

## 项目安装

首先，您已准备好启动 iflearner 服务端和客户端。您可以简单地在终端中启动服务端，如下所示：。
```shell
python3 server.py -n 2
```
> -n：参数n代表接受n个客户端连接，可以修改

现在您已准备好启动将参与学习的 iflearner 客户端。为此，只需再打开两个终端窗口并运行以下命令。

在第一个终端启动客户端1：

```shell
python3 quickstart_mxnet.py --name "client1" --epochs 10
```

在第二个终端启动客户端2：

```shell
python3 quickstart_mxnet.py --name "client2" --epochs 10
```

当然你也可以快速修改脚本`run_server.sh`和`run_client.sh`，然后使用下面的命令快速启动一个体验demo。

在第一个终端启动服务端。
```shell
bash run_server.sh
```

在第二个终端中启动多个客户端。
```shell
bash run_client.sh
```