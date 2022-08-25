import warnings
from typing import Any, List, Tuple, Union

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from iflearner.business.homo import train_client
from iflearner.business.homo.argument import parser
from iflearner.business.homo.train_client import Trainer

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


class Mnist(train_client.SklearnTrainer):
    def __init__(self):
        self._model: LogisticRegression = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
        )
        self._set_initial_params()
        super().__init__(model=self._model)

        (x_train, y_train), (x_test, y_test) = self._load_data()
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def _set_initial_params(self):
        n_classes = 10  # MNIST has 10 classes
        n_features = 784  # Number of features in dataset
        self._model.classes_ = np.array([i for i in range(10)])

        self._model.coef_ = np.zeros((n_classes, n_features))
        if self._model.fit_intercept:
            self._model.intercept_ = np.zeros((n_classes,))

    @staticmethod
    def _load_data() -> Dataset:
        """Loads the MNIST dataset using OpenML.

        OpenML dataset link: https://www.openml.org/d/554
        """
        # mnist_openml = openml.datasets.get_dataset(554)
        # Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
        # X = Xy[:, :-1]  # the last column contains labels
        # y = Xy[:, -1]
        # # First 60000 samples consist of the train set
        # x_train, y_train = X[:60000], y[:60000]
        # x_test, y_test = X[60000:], y[60000:]
        # return (x_train, y_train), (x_test, y_test)
        train_sample = 5000
        x, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, data_home="./data", cache=True
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_sample, test_size=10000
        )
        scaler = StandardScaler()  # 对数据进行归一化，即对数据求方差与均值
        x_train = scaler.fit_transform(x_train)  # 先进行拟合，再进行归一化变换
        x_test = scaler.transform(x_test)  # 只是进行归一化变换
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def _shuffle(x: np.ndarray, y: np.ndarray) -> XY:
        """Shuffle X and y."""
        rng = np.random.default_rng()
        idx = rng.permutation(len(x))
        return x[idx], y[idx]

    @staticmethod
    def _partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
        """Split X and y into a number of partitions."""
        return list(
            zip(np.array_split(x, num_partitions), np.array_split(y, num_partitions))
        )

    def get(self, param_type=Trainer.ParameterType.ParameterModel) -> dict:
        parametes: dict = dict()
        if self._model.fit_intercept:
            parametes["coef"] = self._model.coef_
            parametes["intercept"] = self._model.intercept_
        else:
            parametes["coef"] = self._model.coef_
        return parametes

    def set(
        self, parameters: dict, param_type=Trainer.ParameterType.ParameterModel
    ) -> None:
        self._model.coef_ = parameters["coef"]
        if self._model.fit_intercept:
            self._model.intercept_ = parameters["intercept"]

    def fit(self, epoch: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(self._x_train, self._y_train)

    def evaluate(self, epoch: int) -> dict:
        loss = log_loss(self._y_test, self._model.predict_proba(self._x_test))
        accuracy = self._model.score(self._x_test, self._y_test)
        print(f"epoch:{epoch} | accuracy:{accuracy} | loss:{loss}")
        return {"loss": loss, "accuracy": accuracy}


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mnist = Mnist()
    controller = train_client.Controller(args, mnist)
    controller.run()
