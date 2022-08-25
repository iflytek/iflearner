## 快速开始 (Keras)

在本教程中，我们将介绍如何在 Keras 框架下使用 IFLeaner 在 MNIST 数据集下完成图像分类联邦训练。

我们这个示例默认是包含了两个客户端和一个服务端。每一轮训练，客户端 负责训练并上传模型 参数到服务端，服务端进行聚合， 
并下发聚合后的全局模型参数给每个客户端，然后每个客户端更新聚合后的 模型参数，这将重复多轮。

首先，我们极其推荐先创建一个python虚拟环境来运行，可以通过virtualenv, pyenv, conda等等虚拟工具。

接下来，我们可以通过下述命令快速安装IFLearner库:
```shell
pip install iflearner
```

另外，因为我们想使用 Keras 来完成在 MNIST 数据上的图像分类任务，我们需要继续安装 Keras 库:
```shell
pip install keras==2.9.0
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
创建一个名叫`quickstart_keras.py`的文件，然后按照下述步骤进行操作:

#### 1. 定义模型网络结构

首先，您需要在keras上定义您自己的网络模型。

```python
from typing import Any, Tuple

from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

 # Another way to build your neural net
model: Any = Sequential(
    [
        Dense(32, input_dim=784),  # 输入值784(28*28) => 输出值32
        Activation("relu"),  # 激励函数 转换成非线性数据
        Dense(10),  # 输出为10个单位的结果
        Activation("softmax"),  # 激励函数 调用softmax进行分类
    ]
)
# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # 学习率lr

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(
    optimizer=rmsprop,  # 加速神经网络
    loss="categorical_crossentropy",  # 损失函数
    metrics=["accuracy"],  # 计算误差或准确率
)
```

#### 2. 继承实现Trainer类

其次，您需要实现您的trainer类，继承`iflearner.business.homo.trainer.Trainer`。该类需要实现四个函数，
它们是`get`、`set`、`fit`和`evaluate`函数。我们还提供了`iflearner.business.homo.keras_trainer.KerasTrainer`类。KerasTrainer从`iflearner.business.homo.trainer.Trainer`继承而来，已经实现了常见的`get`和`set`函数。

您可以继承实现该类如下所示:

```python
class Mnist(KerasTrainer):
    def __init__(self):
        # Another way to build your neural net
        model: Any = Sequential(
            [
                Dense(32, input_dim=784),  # 输入值784(28*28) => 输出值32
                Activation("relu"),  # 激励函数 转换成非线性数据
                Dense(10),  # 输出为10个单位的结果
                Activation("softmax"),  # 激励函数 调用softmax进行分类
            ]
        )

        # Another way to define your optimizer
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # 学习率lr

        # We add metrics to get more results you want to see
        # 激活神经网络
        model.compile(
            optimizer=rmsprop,  # 加速神经网络
            loss="categorical_crossentropy",  # 损失函数
            metrics=["accuracy"],  # 计算误差或准确率
        )
        self._model = model
        super(Mnist, self).__init__(model=model)

        (x_train, y_train), (x_test, y_test) = self._load_data()
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    @staticmethod
    def _load_data() -> Dataset:
        # 下载MNIST数据
        # X shape(60000, 28*28) y shape(10000, )
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 数据预处理
        x_train = x_train.reshape(x_train.shape[0], -1) / 255  # normalize
        x_test = x_test.reshape(x_test.shape[0], -1) / 255  # normalize

        # 将类向量转化为类矩阵  数字 5 转换为 0 0 0 0 0 1 0 0 0 0 矩阵
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)
        return (x_train, y_train), (x_test, y_test)

    def fit(self, epoch: int):
        self._model.fit(self._x_train, self._y_train, epochs=1, batch_size=32)

    def evaluate(self, epoch: int) -> dict:
        loss, accuracy = self._model.evaluate(self._x_test, self._y_test)
        print(f"epoch:{epoch} | accuracy:{accuracy} | loss:{loss}")
        return {"loss": loss, "accuracy": accuracy}
```

#### 3. 启动Iflearner的客户端

最后，您需要编写一个`main`函数来启动客户端。

您可以按以下方式执行:

```python
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
python quickstart_keras.py --name client01 --epochs 2
```

打开另一个终端，并且启动第二个客户端进程:
```shell
python quickstart_keras.py --name client02 --epochs 2
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
(iflearner) yiyezhiqiu:quickstart_mxnet lucky$ cd ../quickstart_keras/
(iflearner) yiyezhiqiu:quickstart_keras lucky$ python quickstart_keras.py --name client01 --epochs 2
Namespace(name='client01', epochs=2, server='localhost:50001', enable_ll=0, peers=None, cert=None)
2022-08-03 18:28:25.569565: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
/Users/lucky/opt/anaconda3/envs/iflearner/lib/python3.9/site-packages/keras/optimizers/optimizer_v2/rmsprop.py:135: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(RMSprop, self).__init__(name, **kwargs)
2022-08-03 18:28:27.137 | INFO     | iflearner.business.homo.train_client:run:90 - register to server
2022-08-03 18:28:27.384 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 246.81410300000016ms
2022-08-03 18:28:27.385 | INFO     | iflearner.business.homo.train_client:run:107 - use strategy: FedAvg
2022-08-03 18:28:27.386 | INFO     | iflearner.business.homo.train_client:run:140 - report client ready
2022-08-03 18:28:27.391 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 4.523920999998765ms
2022-08-03 18:28:54.529 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.4368589999946835ms
2022-08-03 18:28:55.466 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
1875/1875 [==============================] - 6s 2ms/step - loss: 0.3668 - accuracy: 0.8968
2022-08-03 18:29:01.852 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
313/313 [==============================] - 1s 2ms/step - loss: 0.2343 - accuracy: 0.9348
epoch:1 | accuracy:0.9348000288009644 | loss:0.23433993756771088
2022-08-03 18:29:02.782 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
2022-08-03 18:29:02.794 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 4.293857000000401ms
2022-08-03 18:29:03.773 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 13.262433999997825ms
2022-08-03 18:29:03.795 | INFO     | iflearner.business.homo.train_client:run:222 - ----- set -----
2022-08-03 18:29:03.797 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.148008999997785ms
2022-08-03 18:29:04.778 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.4469219999995175ms
2022-08-03 18:29:04.800 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2399 - accuracy: 0.9317
2022-08-03 18:29:09.112 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
313/313 [==============================] - 0s 2ms/step - loss: 0.1856 - accuracy: 0.9448
epoch:2 | accuracy:0.9448000192642212 | loss:0.18558283150196075
2022-08-03 18:29:09.686 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
label: FT, points: ([1, 2], [0.23433993756771088, 0.18558283150196075])
label: LT, points: ([1], [0.23433993756771088])
label: FT, points: ([1, 2], [0.9348000288009644, 0.9448000192642212])
label: LT, points: ([1], [0.9348000288009644])
2022-08-03 18:29:10.482 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.3958279999997103ms
```
恭喜您！您已成功构建并运行了您的第一个联邦学习系统。完整的该示例的源代码
参考[Quickstart_Keras](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_keras)。