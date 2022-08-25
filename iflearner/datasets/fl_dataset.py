import numpy as np


class FLDateset:
    def __init__(self) -> None:
        self.train_x = None
        self.test_x = None
        self.test_targets = None
        self.train_targets = None

    def download(self):
        pass

    @property
    def train_labels(self):
        return self.train_targets

    @property
    def test_labels(self):
        return self.test_targets

    @property
    def train_data(self):
        return self.train_x

    @property
    def test_data(self):
        return self.test_x
