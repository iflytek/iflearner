## 快速开始 (TensorFlow)

在本教程中，我们将介绍如何在 Tensorflow 框架下使用 IFLeaner 在 MNIST 数据集下完成图像分类联邦训练。

我们这个示例默认是包含了两个客户端和一个服务端。每一轮训练，客户端 负责训练并上传模型 参数到服务端，服务端进行聚合， 
并下发聚合后的全局模型参数给每个客户端，然后每个客户端更新聚合后的 模型参数，这将重复多轮。

首先，我们极其推荐先创建一个python虚拟环境来运行，可以通过virtualenv, pyenv, conda等等虚拟工具。

接下来，我们可以通过下述命令快速安装IFLearner库:
```shell
pip install iflearner
```

另外，因为我们想使用 Tensorflow 来完成在 MNIST 数据上的图像分类任务，我们需要继续安装 Tensorflow 库:
```shell
pip install tensorflow==2.9.1
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
创建一个名叫`quickstart_tensorflow.py`的文件，然后按照下述步骤进行操作:
#### 1. 定义模型网络结构

首先，您需要在keras上定义您自己的网络模型。

```python
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
```

#### 2. 继承实现Trainer类

其次，您需要实现您的trainer类，继承`iflearner.business.homo.trainer.Trainer`。该类需要实现四个函数，
它们是`get`、`set`、`fit`和`evaluate`函数。我们还提供了`iflearner.business.homo.tensorflow_trainer.TensorFlowTrainer`类。TensorFlowTrainer`iflearner.business.homo.trainer.Trainer`继承而来，已经实现了常见的`get`和`set`函数。

您可以继承实现该类如下所示:

```python
import tensorflow as tf
from iflearner.business.homo.tensorflow_trainer import TensorFlowTrainer
from iflearner.business.homo.train_client import Controller

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

class Mnist(TensorFlowTrainer):
    def __init__(self, model) -> None:
        super().__init__(model)

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    def fit(self, epoch):
        for images, labels in train_ds:
            # train_step(images, labels)
            self._fit(images, labels)           

    @tf.function
    def _fit(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def evaluate(self, epoch):
        for test_images, test_labels in test_ds:
            # test_step(test_images, test_labels)
            self._evaluate(test_images, test_labels)

        print(
            f'Epoch {epoch}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
        return {'Accuracy': train_accuracy.result() * 100}

    @tf.function
    def _evaluate(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
```

#### 3. 启动Iflearner的客户端

最后，您需要编写一个`main`函数来启动客户端。

您可以按以下方式执行:

```python
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model = MyModel()
    mnist = Mnist(model)
    controller = Controller(args, mnist)
    controller.run()
```

在`main`函数中，您需要从`iflearner.business.homo`导入`parser`， 然后调用`parser.parse_args`， 因为我们提供了一些需要解析的常见参数。
如果您自己添加其他参数，可以调用`parser.add_argument`将其添加到`parser.parse_args`之前。在解析参数后，您可以基于之前实现的类创建trainer实例，并将其与`args`传递到`train_client.Controller`函数中。最后，你只需要调用
`controller.run`来启动你的客户端进程。

您可以通过下述命令来启动您的第一个客户端进程:
```shell
python quickstart_tensorflow.py --name client01 --epochs 2
```

打开另一个终端，并且启动第二个客户端进程:
```shell
python quickstart_tensorflow.py --name client02 --epochs 2
```

两个客户端都就绪并启动后，我们可以在任意一个客户端终端上看到类似下述的日志信息:
```text
2022-08-03 18:07:07.406604: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Namespace(name='client01', epochs=2, server='localhost:50001', enable_ll=0, peers=None, cert=None)
2022-08-03 18:07:07.456 | INFO     | iflearner.business.homo.train_client:run:90 - register to server
2022-08-03 18:07:07.479 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 22.46353199999973ms
2022-08-03 18:07:07.479 | INFO     | iflearner.business.homo.train_client:run:107 - use strategy: FedAvg
2022-08-03 18:07:07.480 | INFO     | iflearner.business.homo.train_client:run:140 - report client ready
2022-08-03 18:07:07.482 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.3502309999999795ms
2022-08-03 18:07:11.500 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.1013739999992112ms
2022-08-03 18:07:12.497 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
2022-08-03 18:08:19.804 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Epoch 1, Loss: 0.1414850801229477, Accuracy: 95.73500061035156, Test Loss: 0.0603780597448349, Test Accuracy: 0.0
2022-08-03 18:08:22.759 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
2022-08-03 18:08:24.130 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 149.0828160000035ms
2022-08-03 18:08:31.445 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 3241.557815999997ms
2022-08-03 18:08:32.446 | INFO     | iflearner.business.homo.train_client:run:222 - ----- set -----
2022-08-03 18:08:32.474 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.617487000004303ms
2022-08-03 18:08:33.469 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 8.709660999997482ms
2022-08-03 18:08:33.479 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
2022-08-03 18:09:46.374 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Epoch 2, Loss: 0.11132627725601196, Accuracy: 96.65583038330078, Test Loss: 0.0679144412279129, Test Accuracy: 0.0
2022-08-03 18:09:49.340 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
label: FT, points: ([1, 2], [<tf.Tensor: shape=(), dtype=float32, numpy=95.735>, <tf.Tensor: shape=(), dtype=float32, numpy=96.65583>])
label: LT, points: ([1], [<tf.Tensor: shape=(), dtype=float32, numpy=95.735>])
2022-08-03 18:09:51.374 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.561914999996361ms
```
恭喜您！您已成功构建并运行了您的第一个联邦学习系统。完整的该示例的源代码
参考[Quickstart_Tensorflow](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_tensorflow)。