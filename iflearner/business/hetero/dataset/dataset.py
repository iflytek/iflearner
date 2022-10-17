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

from typing import List
from sklearn.datasets import load_breast_cancer
import csv
import numpy as np


class Dataset:
    features: List[str] = None
    data: List = None
    label: List = None


class Loader:
    """The loader is responsible for reading the csv file and splitting the dataset.

    Attributes:
        complete_dataset (Dataset): The complete dataset, not split.
        training_dataset (Dataset): Training dataset.
        validation_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
    """

    complete_dataset: Dataset = None
    training_dataset: Dataset = None
    validation_dataset: Dataset = None
    test_dataset: Dataset = None

    def __init__(self, csv_path: str, id_index: int = 1, label_index: int = -1) -> None:
        with open(csv_path) as f:
            reader = csv.reader(f)
            temp = next(reader)
        
            features = ""
            data = []
            label = []

            for i, ir in enumerate(reader):
                data[i] = np.asarray(ir[:-1], dtype=np.float64)
                target[i] = np.asarray(ir[-1], dtype=int)

    def split_into_two_datasets(self):
        pass

    def split_into_three_datasets(self):
        pass


if __name__ == "__main__":
    breast = load_breast_cancer()
    loader = Loader("data.csv")
