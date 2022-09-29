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
python3 server.py --addr "127.0.0.1:20001" 
```
> --addr：参数addr代表服务端的地址，可以修改

现在您已准备好启动将参与学习的 iflearner 客户端。为此，只需再打开两个终端窗口并运行以下命令。

在第一个终端启动客户端1：

```shell
python3 client_service.py --name "client1" --server "127.0.0.1:20001"  
--addr "127.0.0.1:10001" --data "examples/mpc/quickstart_piss/piss_data_test.csv"
```
> --name 自身的party_name,在server端不能重复，--server代表服务端的地址，--addr自身的地址，--data代表数据存储路径

在第二个终端启动客户端2：

```shell
python3 client_service.py --name "client2" --server "127.0.0.1:20001"  
--addr "127.0.0.1:10002" --data "examples/mpc/quickstart_piss/piss_data_test.csv"
```

在第三个终端启动客户端3：

```shell
python3 client_service.py --name "client3" --server "127.0.0.1:20001"  
--addr "127.0.0.1:10003"  --data "examples/mpc/quickstart_piss/piss_data_test.csv"
```


新开一个终端，启动查询：
```shell
python3 quickstart_piss.py --param {'10001':'Age', '10002':'Money'} --server "127.0.0.1:10001"
```
> --server 是client_service的服务地址