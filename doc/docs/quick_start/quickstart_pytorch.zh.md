## 快速开始 (PyTorch)

在本教程中，我们将介绍如何在 PyTorch 框架下使用 IFLeaner 在 MNIST 数据集下完成图像分类联邦训练。

我们这个示例默认是包含了两个客户端和一个服务端。每一轮训练，客户端 负责训练并上传模型 参数到服务端，服务端进行聚合， 
并下发聚合后的全局模型参数给每个客户端，然后每个客户端更新聚合后的 模型参数，这将重复多轮。

首先，我们极其推荐先创建一个python虚拟环境来运行，可以通过virtualenv, pyenv, conda等等虚拟工具。

接下来，我们可以通过下述命令快速安装IFLearner库:
```shell
pip install iflearner
```

另外，因为我们想使用 PyTorch 来完成在 MNIST 数据上的图像分类任务，我们需要继续安装 PyTorch 和 torchvision 库:
```shell
pip install torch==1.7.1 torchvision==0.8.2
```

### Ifleaner Server


1.  创建一个名叫 `server.py`的新文件, 引入iflearner库:
```python
from iflearner.business.homo.aggregate_server import main

if __name__ == "__main__":
    main()
```

2. 您可以通过下述命令启动Server进程:
```shell
python server.py -n 2
```
> -n 2: 接收两个客户端进行联邦训练

### Ifleaner Client
创建一个名叫`quickstart_pytorch.py`的文件，然后按照下述步骤进行操作:
#### 1. 定义模型网络结构

首先，您需要在keras上定义您自己的网络模型。

```python
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

#### 2. 继承实现Trainer类

其次，您需要实现您的trainer类，继承`iflearner.business.homo.trainer.Trainer`。该类需要实现四个函数，
它们是`get`、`set`、`fit`和`evaluate`函数。我们还提供了`iflearner.business.homo.pytorch_trainer.PyTorchTrainer`类。PyTorchTrainer从`iflearner.business.homo.trainer.Trainer`继承而来，已经实现了常见的`get`和`set`函数。

您可以继承实现该类如下所示:

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from iflearner.business.homo.train_client import Controller
from iflearner.business.homo.pytorch_trainer import PyTorchTrainer

class Mnist(PyTorchTrainer):
    def __init__(self, lr=0.15, momentum=0.5) -> None:
        self._lr = lr
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'device: {self._device}')
        self._model = Model(num_channels=1, num_classes=10).to(self._device)

        super().__init__(self._model)

        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._loss = F.nll_loss

        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST("./data", train=False, download=True, transform=apply_transform)
        self._train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self._test_data = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    def fit(self, epoch):
        self._model.to(self._device)
        self._model.train()
        print(f"Epoch: {epoch}, the size of training dataset: {len(self._train_data.dataset)}, batch size: {len(self._train_data)}")
        for batch_idx, (data, target) in enumerate(self._train_data):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._loss(output, target)
            loss.backward()
            self._optimizer.step()

    def evaluate(self, epoch):
        self._model.to(self._device)
        self._model.eval()
        test_loss = 0
        correct = 0
        print(f"The size of testing dataset: {len(self._test_data.dataset)}")
        with torch.no_grad():
            for data, target in self._test_data:
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                test_loss += self._loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self._test_data.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(self._test_data.dataset),
            1.   * correct / len(self._test_data.dataset))) 

        return {'loss': test_loss, 'acc': correct}
 ```

#### 3. 启动Iflearner的客户端

最后，您需要编写一个`main`函数来启动客户端。

您可以按以下方式执行:

```python

from iflearner.business.homo.argument import parser

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    mnist = Mnist()
    controller = Controller(args, mnist)
    controller.run()
```

在`main`函数中，您需要从`iflearner.business.homo`导入`parser`， 然后调用`parser.parse_args`， 因为我们提供了一些需要解析的常见参数。
如果您自己添加其他参数，可以调用`parser.add_argument`将其添加到`parser.parse_args`之前。在解析参数后，您可以基于之前实现的类创建trainer实例，并将其与`args`传递到`train_client.Controller`函数中。最后，你只需要调用
`controller.run`来启动你的客户端进程。

您可以通过下述命令来启动您的第一个客户端进程:
```shell
python quickstart_pytorch.py --name client01 --epochs 2
```

打开另一个终端，并且启动第二个客户端进程:
```shell
python quickstart_pytorch.py --name client02 --epochs 2
```

两个客户端都就绪并启动后，我们可以在任意一个客户端终端上看到类似下述的日志信息:
```text
Namespace(name='client01', epochs=2, server='localhost:50001', enable_ll=0, peers=None, cert=None)
device: cpu
2022-08-03 17:33:49.148 | INFO     | iflearner.business.homo.train_client:run:90 - register to server
2022-08-03 17:33:49.165 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 16.620276000000047ms
2022-08-03 17:33:49.166 | INFO     | iflearner.business.homo.train_client:run:107 - use strategy: FedAvg
2022-08-03 17:33:49.166 | INFO     | iflearner.business.homo.train_client:run:140 - report client ready
2022-08-03 17:33:49.170 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 2.700055999999895ms
2022-08-03 17:33:54.192 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.3457160000003299ms
2022-08-03 17:33:55.188 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
Epoch: 1, the size of training dataset: 60000, batch size: 938
2022-08-03 17:34:43.583 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.1023, Accuracy: 9696/10000 (96.96%)
2022-08-03 17:34:48.140 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
2022-08-03 17:34:48.354 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 210.4355039999959ms
2022-08-03 17:34:48.426 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 15.4244869999971ms
2022-08-03 17:34:49.359 | INFO     | iflearner.business.homo.train_client:run:222 - ----- set -----
2022-08-03 17:34:49.362 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.6616899999988277ms
2022-08-03 17:34:50.437 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.947831000002509ms
2022-08-03 17:34:51.367 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
Epoch: 2, the size of training dataset: 60000, batch size: 938
2022-08-03 17:35:38.518 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0833, Accuracy: 9758/10000 (97.58%)
2022-08-03 17:35:43.808 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
label: FT, points: ([1, 2], [0.10231972066191956, 0.08325759547855704])
label: LT, points: ([1], [0.10231972066191956])
label: FT, points: ([1, 2], [0.9696, 0.9758])
label: LT, points: ([1], [0.9696])
2022-08-03 17:35:44.596 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.4475409999903377ms
```
恭喜您！您已成功构建并运行了您的第一个联邦学习系统。完整的该示例的源代码参考[Quickstart_Pytorch](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_pytorch)。