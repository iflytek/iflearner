## Quickstart (Keras)

In this tutorial, we will describe how to use Ifleaner in the Keras framework to complete image classification federated training on the MNIST dataset.

our example contains two clients and one server by default. In each round of training, 
the client is responsible for training and uploading the model parameters to the server, the server
aggregates, and sends the aggregated global model parameters to each client, and then each client 
updates the aggregated model parameters. Multiple rounds will be repeated.

First of all, we highly recommend to create a python virtual environment to run, you can use virtual tools such as virtualenv, pyenv, conda, etc.

Next, we can quickly install the IFLearner library with the following command:
```shell
pip install iflearner
````

Also, since we want to use Keras for image classification tasks on MNIST data, we need to go ahead and install the Keras libraries:
```shell
pip install keras==2.9.0
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

Create a new file named `quickstart_keras.py` and do the following.

#### 1. Define Model Network

Firstly, you need define your model network by using Keras.

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

#### 2. Implement Trainer Class

Secondly, you need implement your trainer class, inheriting from the `iflearner.business.homo.trainer.Trainer` class. The class need to implement four functions,
which are `get`, `set`, `fit` and `evaluate`. We also have provided a `iflearner.business.homo.keras_trainer.KerasTrainer` inheriting from the `iflearner.business.homo.trainer.Trainer` class, 
which has implement usual `get` and `set` functions.

You can use this class as the follow:

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

#### 3. Start Ifleaner Client

Lastly, you need to write a `main` function to start your client.

You can do it as the follow:

```python
if __name__ == '__main__':
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
python quickstart_keras.py --name client01 --epochs 2
```

Open another terminal and start the second client:
```shell
python quickstart_keras.py --name client02 --epochs 2
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
congratulations! You have successfully built and run your first federated learning system. The complete source code 
reference for this example [Quickstart_Tensorflow](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_tensorflow).