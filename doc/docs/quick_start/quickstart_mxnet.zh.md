## 快速开始 (Mxnet)

在本教程中，我们将介绍如何在 Mxnet 框架下使用 IFLeaner 在 MNIST 数据集下完成图像分类联邦训练。

我们这个示例默认是包含了两个客户端和一个服务端。每一轮训练，客户端 负责训练并上传模型 参数到服务端，服务端进行聚合， 
并下发聚合后的全局模型参数给每个客户端，然后每个客户端更新聚合后的 模型参数，这将重复多轮。

首先，我们极其推荐先创建一个python虚拟环境来运行，可以通过virtualenv, pyenv, conda等等虚拟工具。

接下来，我们可以通过下述命令快速安装IFLearner库:
```shell
pip install iflearner
```

另外，因为我们想使用 Mxnet 来完成在 MNIST 数据上的图像分类任务，我们需要继续安装 Mxnet 库:
```shell
pip install mxnet==1.9.1
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
创建一个名叫`quickstart_mxnet.py`的文件，然后按照下述步骤进行操作:

#### 1. 定义模型网络结构

首先，您需要在keras上定义您自己的网络模型。

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

#### 2. 继承实现Trainer类

其次，您需要实现您的trainer类，继承`iflearner.business.homo.trainer.Trainer`。该类需要实现四个函数，
它们是`get`、`set`、`fit`和`evaluate`函数。我们还提供了`iflearner.business.homo.mxnet_trainer.MxnetTrainer`类。MxnetTrainer`iflearner.business.homo.trainer.Trainer`继承而来，已经实现了常见的`get`和`set`函数。

您可以继承实现该类如下所示:

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

#### 3. 启动Iflearner的客户端

最后，您需要编写一个`main`函数来启动客户端。

您可以按以下方式执行:

```python

from iflearner.business.homo.argument import parser

if __name__ == "__main__":
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
python quickstart_mxnet.py --name client01 --epochs 2
```

打开另一个终端，并且启动第二个客户端进程:
```shell
python quickstart_mxnet.py --name client02 --epochs 2
```

两个客户端都就绪并启动后，我们可以在任意一个客户端终端上看到类似下述的日志信息:
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
恭喜您！您已成功构建并运行了您的第一个联邦学习系统。完整的该示例的源代码
参考[Quickstart_Mxnet](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_mxnet)。