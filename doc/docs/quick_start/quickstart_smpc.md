## Quickstart (SMPC)

In this tutorial, we will describe how to use IFLeaner to complete image classification federated training under the MNIST dataset with SMPC.

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
python quickstart_pytorch.py --name client01 --epochs 2 --server "0.0.0.0:50001" --peers "0.0.0.0:50012;0.0.0.0:50013"
```
> Configure peers to use smpc, peers are configured as the listening address of all clients, and the first address is the listening address of the client

Open another terminal and start the second client:
```shell
python quickstart_pytorch.py --name client02 --epochs 2  --server "0.0.0.0:50001" --peers "0.0.0.0:50013;0.0.0.0:50012"
```
> Configure peers to use smpc, peers are configured as the listening address of all clients, and the first address is the listening address of the client

After both clients are ready and started, we can see log messages similar to the following on either client terminal:
```text
Namespace(name='client1', epochs=10, server='0.0.0.0:50001', enable_ll=0, peers='0.0.0.0:50012;0.0.0.0:50013', cert=None)
device: cpu
2022-08-08 19:39:37.971 | INFO     | iflearner.business.homo.train_client:run:89 - register to server
2022-08-08 19:39:37.976 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 4.347216000000209ms
2022-08-08 19:39:37.977 | INFO     | iflearner.business.homo.train_client:run:106 - use strategy: FedAvg
2022-08-08 19:39:37.980 | INFO     | iflearner.communication.peer.peer_client:get_DH_public_key:43 - Public key: b'\x01\x01\xd0\xf2\xa5\xc0-\x7f\x1b\x88\xcb\xc8\\\x91Ra,4\n\xd4]\x97\x99zs\xae7\x1cK]]\x0c\x06\x85\xa1\xb5\x82\x03.\x9a\xe0m\xa3>#\xf7(\xb3x\x89m\xfa\xfbu\x9ca\x95\xf4\x80GA\xd8z\x8fKs\xe0\x98\xe3\x7fX.\xe2Ej\x04c\x08\xcf\xdeF\'\xcc("@q[\xa5\xdf\xb4#\x1c\xd6\xd8\xd1\x05?\x06tO\xfa~Z\x12\x14\x1e\xba\xbe\xaa\xe5/}\xb1Y\xde]\xd8\\\x17\x9cE\xf3Z\xae(\xbfDsf'
2022-08-08 19:39:37.983 | INFO     | iflearner.business.homo.train_client:do_smpc:73 - secret: 122099796455175621216112096188958830464477667871351715488133066776583905683428683953037290811910449486061004359077608812182806437003480898524406977052511968565063977753455848242565116696939311359584774021461025748485006418803112297959930199282888061745382056940467943791248215701671956530776589875636959329985, type: <class 'str'>
2022-08-08 19:39:37.988 | INFO     | iflearner.communication.peer.peer_client:get_SMPC_random_key:56 - Random float: 0.6339090663897411
2022-08-08 19:39:37.988 | INFO     | iflearner.business.homo.train_client:do_smpc:77 - random value: 0.6339090663897411
2022-08-08 19:39:48.296 | INFO     | iflearner.communication.peer.peer_server:send:46 - IN: party: client2, message type: msg_dh_public_key
2022-08-08 19:39:48.298 | INFO     | iflearner.communication.peer.peer_server:send:46 - IN: party: client2, message type: msg_smpc_random_key
2022-08-08 19:39:48.298 | INFO     | iflearner.communication.peer.peer_server:send:56 - Party: client2, Random float: 0.3922334767649399
2022-08-08 19:39:49.023 | INFO     | iflearner.business.homo.train_client:do_smpc:80 - sum all random values: 0.24167558962480118
2022-08-08 19:39:49.025 | INFO     | iflearner.business.homo.train_client:run:139 - report client ready
2022-08-08 19:39:49.027 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 2.0064270000013096ms
2022-08-08 19:39:50.030 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.7326500000010014ms
2022-08-08 19:39:51.033 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 1, the size of training dataset: 60000, batch size: 938
2022-08-08 19:40:20.664 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.1058, Accuracy: 9694/10000 (96.94%)
2022-08-08 19:40:23.830 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:40:23.858 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 17.50900600000449ms
2022-08-08 19:40:24.168 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 10.504099000002043ms
2022-08-08 19:40:24.862 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:40:24.865 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.2220939999991742ms
2022-08-08 19:40:25.173 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.2624359999975354ms
2022-08-08 19:40:25.871 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 2, the size of training dataset: 60000, batch size: 938
2022-08-08 19:41:00.230 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0998, Accuracy: 9726/10000 (97.26%)
2022-08-08 19:41:03.992 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:41:04.020 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 17.794709000000353ms
2022-08-08 19:41:04.367 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 9.483094000003689ms
2022-08-08 19:41:05.024 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:41:05.027 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.9964000000101123ms
2022-08-08 19:41:05.374 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.6533630000026278ms
2022-08-08 19:41:06.032 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 3, the size of training dataset: 60000, batch size: 938
2022-08-08 19:41:33.934 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0871, Accuracy: 9743/10000 (97.43%)
2022-08-08 19:41:37.425 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:41:37.492 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 18.108728000001406ms
2022-08-08 19:41:37.514 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 8.546628999994255ms
2022-08-08 19:41:38.496 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:41:38.498 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.2796400000070207ms
2022-08-08 19:41:38.519 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 0.9813249999979234ms
2022-08-08 19:41:39.503 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 4, the size of training dataset: 60000, batch size: 938
2022-08-08 19:42:12.085 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0989, Accuracy: 9705/10000 (97.05%)
2022-08-08 19:42:15.499 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:42:15.513 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 2.581355999978996ms
2022-08-08 19:42:15.694 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 8.077353999993875ms
2022-08-08 19:42:16.517 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:42:16.519 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.3156910000020616ms
2022-08-08 19:42:16.701 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 2.0472559999973328ms
2022-08-08 19:42:17.524 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 5, the size of training dataset: 60000, batch size: 938
2022-08-08 19:42:53.300 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0639, Accuracy: 9806/10000 (98.06%)
2022-08-08 19:42:56.654 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:42:56.667 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 1.813790000028348ms
2022-08-08 19:42:56.892 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 7.262933999982124ms
2022-08-08 19:42:57.672 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:42:57.675 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.5553380000028483ms
2022-08-08 19:42:57.898 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.6520599999978458ms
2022-08-08 19:42:58.679 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 6, the size of training dataset: 60000, batch size: 938
2022-08-08 19:43:26.257 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0753, Accuracy: 9787/10000 (97.87%)
2022-08-08 19:43:30.128 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:43:30.143 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 2.0744939999985945ms
2022-08-08 19:43:31.048 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 9.83907899998826ms
2022-08-08 19:43:31.148 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:43:31.151 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 2.054303000022628ms
2022-08-08 19:43:32.055 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.705947000004926ms
2022-08-08 19:43:32.153 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 7, the size of training dataset: 60000, batch size: 938
2022-08-08 19:43:58.396 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0736, Accuracy: 9768/10000 (97.68%)
2022-08-08 19:44:01.113 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:44:01.125 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 1.890878999972756ms
2022-08-08 19:44:02.184 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 9.515048000025672ms
2022-08-08 19:44:03.132 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:44:03.135 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.91251899997269ms
2022-08-08 19:44:03.188 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.6274970000154099ms
2022-08-08 19:44:04.140 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 8, the size of training dataset: 60000, batch size: 938
2022-08-08 19:44:34.161 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0646, Accuracy: 9801/10000 (98.01%)
2022-08-08 19:44:37.132 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:44:37.151 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 12.143503999993754ms
2022-08-08 19:44:37.328 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 7.2212239999771555ms
2022-08-08 19:44:38.153 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:44:38.156 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.8954930000063541ms
2022-08-08 19:44:38.335 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 2.6670950000493576ms
2022-08-08 19:44:39.161 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 9, the size of training dataset: 60000, batch size: 938
2022-08-08 19:45:04.166 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0694, Accuracy: 9781/10000 (97.81%)
2022-08-08 19:45:06.821 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
2022-08-08 19:45:06.841 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 12.79627700000674ms
2022-08-08 19:45:07.453 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 9.497581000005084ms
2022-08-08 19:45:07.846 | INFO     | iflearner.business.homo.train_client:run:221 - ----- set -----
2022-08-08 19:45:07.848 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.1865850000276623ms
2022-08-08 19:45:09.465 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.6427019999696313ms
2022-08-08 19:45:09.858 | INFO     | iflearner.business.homo.train_client:run:149 - ----- fit <FT> -----
Epoch: 10, the size of training dataset: 60000, batch size: 938
2022-08-08 19:45:48.696 | INFO     | iflearner.business.homo.train_client:run:167 - ----- evaluate <FT> -----
The size of testing dataset: 10000
Test set: Average loss: 0.0615, Accuracy: 9814/10000 (98.14%)
2022-08-08 19:45:53.514 | INFO     | iflearner.business.homo.train_client:run:178 - ----- get <FT> -----
label: FT, points: ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.10584523560330272, 0.09984834497272968, 0.0871084996862337, 0.09891319364532829, 0.063905619766374, 0.07528823107918725, 0.07361029261836957, 0.06460160582875542, 0.0694242621988058, 0.06149101790403947])
label: LT, points: ([1], [0.10584523560330272])
label: FT, points: ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.9694, 0.9726, 0.9743, 0.9705, 0.9806, 0.9787, 0.9768, 0.9801, 0.9781, 0.9814])
label: LT, points: ([1], [0.9694])
2022-08-08 19:45:54.253 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 2.060518000007505ms
```
congratulations! You have successfully built and run your first federated learning system. The complete source code reference for this example [Quickstart_SMPC](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_smpc).