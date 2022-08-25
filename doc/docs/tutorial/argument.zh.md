## 启动参数

### 客户端
您需要从`iflearner.business.homo`导入`parser`， 然后调用`parser.parse_args`， 因为我们提供了一些需要解析的常见参数。
如果您自己添加其他参数，可以调用`parser.add_argument`将其添加到`parser.parse_args`之前。在解析参数后，您可以基于之前实现的类创建trainer实例，并将其与`args`传递到`train_client.Controller`函数中。最后，你只需要调用
`controller.run`来启动你的客户端进程。

下面是客户端默认的参数:

| option | type |  describe | default |
| :-----| :---- | :---- |:----: |
| name | str | 客户端名称(必须唯一)  | client |
| epochs | int | 总的训练轮数 | 10 |
| server | str | 链接聚合服务端的地址 | localhost:50001 |
| enable-ll | int | 启动本地训练进行对比 (1 、 0), 1代表开启 | 0 |
| peers | str | 如果参数已指定，则启用 SMPC。 所有客户端的地址并使用分号分隔所有地址。 第一个是你自己的地址。 |  |
| cert | str | 服务器 SSL 证书的路径。 如果指定，则使用安全通道连接到服务器|  |

### 服务端

服务端参数列表如下:

| 选项 | 类型 |  描述 | 默认值 |
| :-----| :---- | :---- |:----: |
| num | int | 客户端数目  | 0 |
| epochs | int | 总的聚合轮数 |  |
| addr | str | 聚合服务端本身监听地址(用于客户端链接) | "0.0.0.0:50001" |
| http_addr | str |联邦训练状态监听地址(用于查看联邦训练状态) | "0.0.0.0:50002" |
| strategy | str | 聚合策略 (FedAvg、Scaffold、FedOpt、qFedAvg、FedNova) | FedAvg |
| strategy_params | dict | 聚合策略参数 | {} |
