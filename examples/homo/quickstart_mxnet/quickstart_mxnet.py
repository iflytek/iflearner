from typing import Any, Tuple

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon, nd
from mxnet.gluon import nn

from iflearner.business.homo.argument import parser
from iflearner.business.homo.mxnet_trainer import MxnetTrainer
from iflearner.business.homo.train_client import Controller


def model():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(64, activation="relu"))
    net.add(nn.Dense(10))
    net.collect_params().initialize()
    return net


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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mnist = Mnist()
    controller = Controller(args, mnist)
    controller.run()
