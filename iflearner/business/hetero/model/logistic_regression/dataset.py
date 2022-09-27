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

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    breast = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        breast.data, breast.target, random_state=1)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    return x_train, y_train, x_test, y_test


def vertically_partition_data(X, X_test, A_idx, B_idx):
    XA = X[:, A_idx]
    XB = X[:, B_idx]
    # print(X.shape[0], np.ones(X.shape[0]))
    # print(X.shape[1], np.ones(X.shape[1]))
    XB = np.c_[np.ones(X.shape[0]), XB]
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]
    XB_test = np.c_[np.ones(XB_test.shape[0]), XB_test]
    return XA, XB, XA_test, XB_test


def get_guest_data():
    x, y, x_test, y_test = load_data()
    XA, XB, XA_test, XB_test = vertically_partition_data(x, x_test, [
                                                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return XB, y


def get_host_data():
    x, y, x_test, y_test = load_data()
    XA, XB, XA_test, XB_test = vertically_partition_data(x, x_test, [
                                                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return XA
