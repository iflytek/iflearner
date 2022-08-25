## Quickstart (PyTorch)

In this tutorial, we will describe how to use Ifleaner in the PyTorch framework to complete image classification federated training on the MNIST dataset.

our example contains two clients and one server by default. In each round of training, 
the client is responsible for training and uploading the model parameters to the server, the server
aggregates, and sends the aggregated global model parameters to each client, and then each client 
updates the aggregated model parameters. Multiple rounds will be repeated.

First of all, we highly recommend to create a python virtual environment to run, you can use virtual tools such as virtualenv, pyenv, conda, etc.

Next, we can quickly install the IFLearner library with the following command:
```shell
pip install iflearner
````

Also, since we want to use PyTorch for image classification tasks on MNIST data, we need to go ahead and install the PyTorch and torchvision libraries:
```shell
pip install torch==1.7.1 torchvision==0.8.2
````

### Ifleaner Server

1.  Create a new file named `server.py`, import iflearner:
```python
from iflearner.business.homo.aggregate_server import main

if __name__ == "__main__":
    main()
```

2. You can start the server with the follow command:
```shell
python server.py -n 2
```
> -n 2: Receive two clients for federated training

### Ifleaner Client
Create a new file named `quickstart_pytorch.py` and do the following.

#### 1. Define Model Network

Firstly, you need define your model network by using PyTorch.

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

#### 2. Implement Trainer Class

Secondly, you need implement your trainer class, inheriting from the `iflearner.business.homo.trainer.Trainer` class. 
The class need to implement four functions, which are `get`, `set`, `fit` and `evaluate`. 
We also have provided a `iflearner.business.homo.pytorch_trainer.PyTorchTrainer` inheriting from the `iflearner.business.homo.trainer.Trainer` class, which has implement usual `get` and `set` functions.

You can use this class as the follow:

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

#### 3. Start Ifleaner Client

Lastly, you need to write a `main` function to start your client. 

You can do it as the follow:

```python

from iflearner.business.homo.argument import parser

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    mnist = Mnist()
    controller = Controller(args, mnist)
    controller.run()
```

In the ``main`` function, you need import `parser` from `iflearner.business.homo.argument` firstly and then call `parser.parse_args`, because we provided some common arguments that
need to be parsered. If you want to add addtional arguments for yourself, you can call `parser.add_argument` repeatedly to add them before `parser.parse_args`
has been called. After parsered arguments, you can create your trainer instance base on previous implemented class, and put it with `args` to `train_client.Controller`. In the end, you just need call `controller.run` to run your client.

You can use follow command to start the first client:
```shell
python quickstart_pytorch.py --name client01 --epochs 2
```

Open another terminal and start the second client:
```shell
python quickstart_pytorch.py --name client02 --epochs 2
```

After both clients are ready and started, we can see log messages similar to the following on either client terminal:
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
congratulations! You have successfully built and run your first federated learning system. The complete source code reference for this example [Quickstart_Pytorch](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_pytorch).