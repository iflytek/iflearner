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
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from .fedopt import FedOpt


class FedAvgm(FedOpt):
    """Implementation based on https://arxiv.org/abs/1909.06335.

    Attributes:
        learning_rate (float, optional): learning rate. Defaults to 1.
        momentum (float, optional): momentum factor. Defaults to 0.0.
    """

    def __init__(
        self,
        learning_rate: float = 1,
        momentum: float = 0.0,
    ) -> None:

        super().__init__(learning_rate=learning_rate)
        self._momentum_vector: Optional[Dict[str, npt.NDArray[np.float32]]] = None
        self._momentum = momentum

    def step(
        self, pseudo_gradient: Dict[str, npt.NDArray[np.float32]]
    ) -> Dict[str, npt.NDArray[np.float32]]:

        """a step to optimize parameters of server model with pseudo gradient.

        Args:
            pseudo_gradient (Dict[str, npt.NDArray[np.float32]]): the pseudo gradient of server model

        Returns:
            Dict[str, npt.NDArray[np.float32]]: parameters of server model after step
        """

        if self._momentum_vector is None:
            self._momentum_vector = dict()
            for key, value in pseudo_gradient.items():
                self._momentum_vector[key] = -value
        else:
            for key, value in pseudo_gradient.items():
                self._momentum_vector[key] = (
                    self._momentum * self._momentum_vector[key] - value
                )
        pseudo_gradient = self._momentum_vector

        new_params = dict()
        for key, value in self._params.items():
            new_params[key] = value.reshape((-1)) - self._lr * pseudo_gradient[key]
            self._params[key] = new_params[key].reshape(value.shape)

        return new_params
