#  Copyright 2022 iFLYTEK. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from iflearner.business.util.metric_dev import *

file_dir = "./client1"


def test_loss_metric() -> None:
    loss_metric = LossMetric(metric_name="test_loss_metric", file_dir=file_dir)
    epochs = [i + 1 for i in range(10)]
    loss = [0.01 * (10 - i) for i in range(10)]
    for element in zip(epochs, loss):
        loss_metric.add(element[0], element[1])
    print(loss_metric)
    loss_metric.plot()

    loss_metric1 = LossMetric(file_dir=file_dir)
    loss_metric1.metric_name = "test_loss_metric1"
    loss_metric1.add(epochs, loss, train_type=TrainType.LocalTrain)  # type: ignore
    print(loss_metric1)
    loss_metric1.plot()


def test_accuracy_metric() -> None:
    accuracy_metric = AccuracyMetric(
        metric_name="test_accuracy_metric", file_dir=file_dir
    )
    epochs = [i + 1 for i in range(10)]
    accuracys = [0.1 * (10 - i) for i in range(10)]
    accuracy_metric.add(x=epochs, y=accuracys)  # type: ignore
    print(accuracy_metric)
    accuracy_metric.plot()


def test_metrics() -> None:
    metrics = Metrics(file_dir=file_dir)

    loss_metric = LossMetric(metric_name="test_metrics_loss")
    epochs = [i + 1 for i in range(10)]
    loss = [0.01 * (10 - i) for i in range(10)]
    for element in zip(epochs, loss):
        loss_metric.add(element[0], element[1])  # type: ignore

    accuracy_metric = AccuracyMetric(metric_name="test_metrics_accuracy")
    epochs = [i + 1 for i in range(10)]
    local_auc = [0.05 * i for i in range(10)]
    fedrated_auc = [0.1 * i for i in range(10)]
    # local train metric
    for element in zip(epochs, local_auc):
        accuracy_metric.add(element[0], element[1], train_type=TrainType.LocalTrain)  # type: ignore
    # federated train metric
    for element in zip(epochs, fedrated_auc):
        accuracy_metric.add(element[0], element[1], train_type=TrainType.FederatedTrain)  # type: ignore

    metrics.add(loss_metric)  # type: ignore
    metrics.add(accuracy_metric)  # type: ignore
    assert len(metrics) == 2

    metrics.plot()
    print(metrics)

    metrics.dump()
    metrics.load()


if __name__ == "__main__":
    test_loss_metric()
    test_accuracy_metric()
    test_metrics()
