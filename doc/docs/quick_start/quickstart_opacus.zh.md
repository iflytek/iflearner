## 快速开始 (Opacus)

在本教程中，将帮助您了解结合差分隐私加密技术， 使用Opacus库，在 pytorch 下运行联邦任务

我们这个示例默认是包含了两个客户端和一个服务端。每一轮训练，客户端 负责训练并上传模型 参数到服务端，服务端进行聚合， 
并下发聚合后的全局模型参数给每个客户端，然后每个客户端更新聚合后的 模型参数，这将重复多轮。

首先，我们极其推荐先创建一个python虚拟环境来运行，可以通过virtualenv, pyenv, conda等等虚拟工具。

接下来，我们可以通过下述命令快速安装IFLearner库:
```shell
pip install iflearner
```

另外，因为我们想使用 PyTorch 来完成在 MNIST 数据上的图像分类任务，我们需要继续安装Opacus、 PyTorch 和 torchvision 库:
```shell
pip install opacus==1.1.3 torch==1.8.1 torchvision==0.9.1
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

同时，我们集成也集成了Opacus差分隐私库。

您可以继承实现该类如下所示:

```python
class Mnist(PyTorchTrainer):
    def __init__(self, lr=0.15, momentum=0.5, delta=1e-5) -> None:
        self._lr = lr
        self._delta = delta
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"device: {self._device}")
        model = Model(num_channels=1, num_classes=10).to(self._device)

        super().__init__(model)

        optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._loss = F.nll_loss

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=apply_transform
        )
        test_dataset = datasets.MNIST(
            "./data", train=False, download=True, transform=apply_transform
        )
        train_data = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        self._test_data = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False
        )
        self._privacy_engine = PrivacyEngine()
        self._model, self._optimizer, self._train_data = self._privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_data,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    def fit(self, epoch):
        self._model.to(self._device)
        self._model.train()
        print(
            f"Epoch: {epoch}, the size of training dataset: {len(self._train_data.dataset)}, batch size: {len(self._train_data)}"
        )
        losses = []
        for batch_idx, (data, target) in enumerate(self._train_data):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._loss(output, target)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

        epsilon, best_alpha = self._privacy_engine.accountant.get_privacy_spent(
            delta=self._delta
        )
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {self._delta}) for α = {best_alpha}"
        )

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
                test_loss += self._loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self._test_data.dataset)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(self._test_data.dataset),
                100.0 * correct / len(self._test_data.dataset),
            )
        )

        return {"loss": test_loss, "acc": correct / len(self._test_data.dataset)}
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
/Users/lucky/opt/anaconda3/envs/iflearner/lib/python3.9/site-packages/opacus/privacy_engine.py:133: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.
  warnings.warn(
2022-08-08 20:54:51.294 | INFO     | iflearner.business.homo.train_client:run:89 - register to server
2022-08-08 20:54:51.308 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 13.392697000000009ms
2022-08-08 20:54:51.308 | INFO     | iflearner.business.homo.train_client:run:106 - use strategy: FedAvg
2022-08-08 20:54:51.309 | INFO     | iflearner.business.homo.train_client:run:139 - report client ready
2022-08-08 20:54:51.311 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.5273650000002803ms
2022-08-08 20:54:53.322 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.9921759999999011ms
2022-08-08 20:54:54.325 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 1, the size of training dataset: 60000, batch size: 938
/Users/lucky/opt/anaconda3/envs/iflearner/lib/python3.9/site-packages/torch/nn/modules/module.py:795: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
  warnings.warn("Using a non-full backward hook when the forward contains multiple autograd Nodes "
Train Epoch: 1  Loss: 1.963445 (ε = 0.53, δ = 1e-05) for α = 16.0
2022-08-08 20:55:49.100 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.6190, Accuracy: 7907/10000 (79.07%)
2022-08-08 20:55:51.779 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 20:55:51.785 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 1.787130000003856ms
2022-08-08 20:55:52.656 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 9.798496000001933ms
2022-08-08 20:55:52.789 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 20:55:52.794 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.5915189999944346ms
2022-08-08 20:55:53.659 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 2.2224549999947385ms
2022-08-08 20:55:53.799 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 2, the size of training dataset: 60000, batch size: 938
Train Epoch: 2  Loss: 1.975834 (ε = 0.55, δ = 1e-05) for α = 16.0
2022-08-08 20:56:41.185 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.6178, Accuracy: 8213/10000 (82.13%)
2022-08-08 20:56:44.589 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
label: FT, points: ([1, 2], [0.6190427913188934, 0.617782280254364])
label: LT, points: ([1], [0.6190427913188934])
label: FT, points: ([1, 2], [0.7907, 0.8213])
label: LT, points: ([1], [0.7907])
2022-08-08 20:56:45.487 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.8401949999997669ms
```
恭喜您！您已成功构建并运行了您的第一个联邦学习系统。完整的该示例的源代码参考[Quickstart_Pytorch](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_opacus)。