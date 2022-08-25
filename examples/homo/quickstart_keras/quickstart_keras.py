from typing import Any, Tuple

import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

from iflearner.business.homo.argument import parser
from iflearner.business.homo.keras_trainer import KerasTrainer
from iflearner.business.homo.train_client import Controller

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]


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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mnist = Mnist()
    controller = Controller(args, mnist)
    controller.run()
