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
import pickle
from enum import Enum, unique
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt

Scalar = Union[bool, float, int, str]


@unique
class TrainType(Enum):
    """define the type of train.

    supported local and federated
    """

    LocalTrain = "Local"
    FederatedTrain = "Federated"


class BaseMetric(object):
    """Base class for metric."""

    def __init__(
        self, metric_name: str, x_label: str, y_label: str, file_dir: str = "./"
    ):
        """
        Args:
            metric_name: metric name
            x_label: x-axis label for drawing
            y_label: y-axis label for drawing
            file_dir: The file path to save metic.
        """
        self._x_label = x_label
        self._y_label = y_label
        self._metric_name = metric_name
        self._local_x_elements: List[Scalar] = []
        self._local_y_elements: List[Scalar] = []
        self._federate_x_elements: List[Scalar] = []
        self._federate_y_elements: List[Scalar] = []
        self._file_dir = file_dir

    def add(
        self,
        x: Union[Scalar, List[Scalar]],
        y: Union[Scalar, List[Scalar]],
        train_type: TrainType = TrainType.FederatedTrain,
    ) -> None:
        """add scalar to elements.

        Args:
            train_type: support localTrain and federatedTrain.
            x: x-axis scalar, for example as `epoch` value
            y: y-axis scalar, for example as `loss` value

        Returns: None
        """
        if train_type == TrainType.LocalTrain:
            if isinstance(x, list):
                self._local_x_elements.extend(x)  # type: ignore
                self._local_y_elements.extend(y)  # type: ignore
            else:
                self._local_x_elements.append(x)  # type: ignore
                self._local_y_elements.append(y)  # type: ignore
        elif train_type == TrainType.FederatedTrain:
            if isinstance(x, list):
                self._federate_x_elements.extend(x)  # type: ignore
                self._federate_y_elements.extend(y)  # type: ignore
            else:
                self._federate_x_elements.append(x)  # type: ignore
                self._federate_y_elements.append(y)  # type: ignore

    @property
    def file_dir(self) -> str:
        return self._file_dir

    @file_dir.setter
    def file_dir(self, file_path: str) -> None:
        self._file_dir = file_path

    @property
    def metric_name(self):
        return self._metric_name

    @metric_name.setter
    def metric_name(self, name: str):
        self._metric_name = name

    @property
    def x_label(self) -> str:
        return self._x_label

    @property
    def y_label(self) -> str:
        return self._y_label

    @property
    def local_x_elements(self) -> List[Scalar]:
        return self._local_x_elements

    @property
    def local_y_elements(self) -> List[Scalar]:
        return self._local_y_elements

    @property
    def federate_x_elements(self) -> List[Scalar]:
        return self._federate_x_elements

    @property
    def federate_y_elements(self) -> List[Scalar]:
        return self._federate_y_elements

    def plot(self):
        plt.clf()
        if len(self.local_x_elements):
            plt.plot(
                self.local_x_elements,
                self.local_y_elements,
                label=TrainType.LocalTrain.value,
            )
        if len(self.federate_x_elements):
            plt.plot(
                self.federate_x_elements,
                self.federate_y_elements,
                label=TrainType.FederatedTrain.value,
            )
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.metric_name)
        plt.legend()
        plt.savefig(f"{self._file_dir}/{self.metric_name}.png")

    def __repr__(self) -> str:
        rep = f"Metric:{self.metric_name}:\n"
        if len(self.local_x_elements):
            rep += "local train mode:\n"
            rep += reduce(
                lambda a, b: a + b,
                [
                    f"\t{self.x_label}:{xy_element[0]} {self.y_label}:{xy_element[1]}\n"
                    for xy_element in zip(self.local_x_elements, self.local_y_elements)
                ],
            )
        if len(self.federate_x_elements):
            rep += "federated train mode:\n"
            rep += reduce(
                lambda a, b: a + b,
                [
                    f"\t{self.x_label}:{xy_element[0]} {self.y_label}:{xy_element[1]}\n"
                    for xy_element in zip(
                        self.federate_x_elements, self.federate_y_elements
                    )
                ],
            )
        return rep

    def __str__(self) -> str:
        return self.__repr__()


class LossMetric(BaseMetric):
    """loss metric class."""

    def __init__(self, metric_name: str = "loss", file_dir: str = ""):
        super().__init__(
            metric_name=metric_name, x_label="epoch", y_label="loss", file_dir=file_dir
        )


class AccuracyMetric(BaseMetric):
    """accuracy metric class."""

    def __init__(self, metric_name: str = "accuracy", file_dir: str = ""):
        super().__init__(
            metric_name=metric_name,
            x_label="epoch",
            y_label="accuracy",
            file_dir=file_dir,
        )


class F1Metric(BaseMetric):
    """f1 metric class."""

    def __init__(self, metric_name: str = "f1", file_dir: str = ""):
        super().__init__(
            metric_name=metric_name, x_label="epoch", y_label="f1", file_dir=file_dir
        )


class Metrics:
    """Statistical metric information, such as loss, accuracy, etc..."""

    def __init__(self, file_dir: str = "./") -> None:
        """
        Args:
            file_dir: The file path to save metic.
        """
        self._figs: Dict[str, Any] = dict()
        self._metrics: List[BaseMetric] = []
        self._file_dir = file_dir
        Path(file_dir).mkdir(parents=True, exist_ok=True)

    @property
    def metrics(self) -> List[BaseMetric]:
        return self._metrics

    def add(self, metric: BaseMetric) -> None:
        """add metric to metrics list.

        Args:
            metric: class Metric, for example as LossMetric.

        Returns: None
        """
        metric.file_dir = self._file_dir
        self._metrics.append(metric)

    def plot(self) -> None:
        """plot and save to file."""
        for metric in self.metrics:
            metric.plot()

    def dump(self) -> None:
        """save metrics data to file."""
        with open(f"{self._file_dir}/metrics.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self):
        """load metrics data from file."""
        with open(f"{self._file_dir}/metrics.pkl", "rb") as f:
            metric = pickle.load(f)
        return metric

    def __len__(self):
        return len(self.metrics)

    def __repr__(self) -> str:
        rep = ""
        for metric in self.metrics:
            rep += str(metric) + "\n"
        return rep

    def __str__(self) -> str:
        return self.__repr__()
