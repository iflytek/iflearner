## Quickstart (Mxnet)

In this tutorial, we will describe how to use Ifleaner in the Mxnet framework to complete image classification federated training on the MNIST dataset.

our example contains two clients and one server by default. In each round of training, 
the client is responsible for training and uploading the model parameters to the server, the server
aggregates, and sends the aggregated global model parameters to each client, and then each client 
updates the aggregated model parameters. Multiple rounds will be repeated.

First of all, we highly recommend to create a python virtual environment to run, you can use virtual tools such as virtualenv, pyenv, conda, etc.

Next, we can quickly install the IFLearner library with the following command:
```shell
pip install iflearner
````

Also, since we want to use Mxnet for image classification tasks on MNIST data, we need to go ahead and install the Mxnet libraries:
```shell
pip install mxnet==1.9.1
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
Create a new file named `quickstart_mxnet.py` and do the following.

#### 1. Define Model Network

Firstly, you need define your model network by using Mxnet.

```python
import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon, nd
from mxnet.gluon import nn

def model():
 net = nn.Sequential()
 net.add(nn.Dense(256, activation="relu"))
 net.add(nn.Dense(64, activation="relu"))
 net.add(nn.Dense(10))
 net.collect_params().initialize()
 return net
```


#### 2. Implement Trainer Class

Secondly, you need implement your trainer class, inheriting from the `iflearner.business.homo.trainer.Trainer` class.
The class need to implement four functions, which are `get`, `set`, `fit` and `evaluate`.
We also have provided a `iflearner.business.homo.mxnet_trainer.MxnetTrainer` inheriting from the `iflearner.business.homo.trainer.Trainer` class, which has implement usual `get` and `set` functions.

You can use this class as the follow:

```python
from typing import Any, Tuple

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon, nd
from mxnet.gluon import nn

from iflearner.business.homo.argument import parser
from iflearner.business.homo.mxnet_trainer import MxnetTrainer
from iflearner.business.homo.train_client import Controller

class Mnist(MxnetTrainer):
    def __init__(self):
        self._model = model()
        init = nd.random.uniform(shape=(2, 784))
        self._model(init)
        super().__init__(model=self._model)
        self._train_data, self._val_data = self._load_data()
        self._DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

    @staticmethod
    def _load_data() -> Tuple[Any, Any]:
        print("Download Dataset")
        mnist = mx.test_utils.get_mnist()
        batch_size = 100
        train_data = mx.io.NDArrayIter(
            mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
        )
        val_data = mx.io.NDArrayIter(
            mnist["test_data"], mnist["test_label"], batch_size
        )
        return train_data, val_data

    def fit(self, epoch: int):
        trainer = gluon.Trainer(
            self._model.collect_params(), "sgd", {"learning_rate": 0.01}
        )
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        self._train_data.reset()
        num_examples = 0
        for batch in self._train_data:
            data = gluon.utils.split_and_load(
                batch.data[0], ctx_list=self._DEVICE, batch_axis=0
            )
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=self._DEVICE, batch_axis=0
            )
            outputs = []
            with ag.record():
                for x, y in zip(data, label):
                    z = self._model(x)
                    loss = softmax_cross_entropy_loss(z, y)
                    loss.backward()
                    outputs.append(z.softmax())
                    num_examples += len(x)
            metrics.update(label, outputs)
            trainer.step(batch.data[0].shape[0])
        trainings_metric = metrics.get_name_value()
        [accuracy, loss] = trainings_metric
        print(f"epoch :{epoch}: accuracy:{float(accuracy[1])} loss：{float(loss[1])}")

    def evaluate(self, epoch: int) -> dict:
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        self._val_data.reset()
        num_examples = 0
        for batch in self._val_data:
            data = gluon.utils.split_and_load(
                batch.data[0], ctx_list=self._DEVICE, batch_axis=0
            )
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=self._DEVICE, batch_axis=0
            )
            outputs = []
            for x in data:
                outputs.append(self._model(x).softmax())
                num_examples += len(x)
            metrics.update(label, outputs)
        metrics.update(label, outputs)
        [accuracy, loss] = metrics.get_name_value()
        print(f"Evaluation accuracy:{accuracy} loss:{loss}")
        return {"loss": float(loss[1]), "accuracy": float(accuracy[1])}
```

#### 3. tart Ifleaner Client

Lastly, you need to write a `main` function to start your client.

You can do it as the follow:

```python

from iflearner.business.homo.argument import parser

if __name__ == "__main__":
args = parser.parse_args()
print(args)
mnist = Mnist()
controller = Controller(args, mnist)
controller.run()
```

In the `main` function, you need import `parser` from `iflearner.business.homo.argument` firstly and then call `parser.parse_args`,
because we provided some common arguments that need to be parsered. If you want to add addtional arguments for yourself, you can call 
`parser.add_argument` repeatedly to add them before `parser.parse_args` has been called. After parsered arguments, you can create your 
trainer instance base on previous implemented class, and put it with `args` to `train_client.Controller`. In the end, you just need call
`controller.run` to run your client.

You can use follow command to start the first client:
```shell
python quickstart_mxnet.py --name client01 --epochs 2
```

Open another terminal and start the second client:
```shell
python quickstart_mxnet.py --name client02 --epochs 2
```

After both clients are ready and started, we can see log messages similar to the following on either client terminal:
```text
Namespace(name='client01', epochs=2, server='localhost:50001', enable_ll=0, peers=None, cert=None)
Download Dataset
2022-08-03 18:20:44.788 | INFO     | iflearner.business.homo.train_client:run:90 - register to server
2022-08-03 18:20:44.827 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 38.734709000000755ms
2022-08-03 18:20:44.830 | INFO     | iflearner.business.homo.train_client:run:107 - use strategy: FedAvg
2022-08-03 18:20:44.832 | INFO     | iflearner.business.homo.train_client:run:140 - report client ready
2022-08-03 18:20:44.836 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 2.204193999999937ms
2022-08-03 18:22:39.393 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.843368999999484ms
2022-08-03 18:22:40.203 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
epoch :1: accuracy:0.4393666666666667 loss：2.1208519152323406
2022-08-03 18:22:45.960 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Evaluation accuracy:('accuracy', 0.7123762376237623) loss:('cross-entropy', 1.6562039970171334)
2022-08-03 18:22:46.386 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
2022-08-03 18:22:46.469 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 17.984263000002443ms
2022-08-03 18:22:47.532 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 79.32561299999463ms
2022-08-03 18:22:48.486 | INFO     | iflearner.business.homo.train_client:run:222 - ----- set -----
2022-08-03 18:22:48.491 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 3.8189600000180235ms
2022-08-03 18:22:48.538 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 4.523907999981702ms
2022-08-03 18:22:49.495 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
epoch :2: accuracy:0.7846166666666666 loss：1.017146420733134
2022-08-03 18:22:54.082 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Evaluation accuracy:('accuracy', 0.8396039603960396) loss:('cross-entropy', 0.633656327464793)
2022-08-03 18:22:54.298 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
label: FT, points: ([1, 2], [1.6562039970171334, 0.633656327464793])
label: LT, points: ([1], [1.6562039970171334])
label: FT, points: ([1, 2], [0.7123762376237623, 0.8396039603960396])
label: LT, points: ([1], [0.7123762376237623])
2022-08-03 18:22:55.326 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.322818999994979ms
```
congratulations! You have successfully built and run your first federated learning system. The complete source code 
reference for this example [Quickstart_Mxnet](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_mxnet).
