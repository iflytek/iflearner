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
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .fedopt import FedOpt


class FedAdam(FedOpt):
    """
    Attributes:
        learning_rate (float, optional): learning rate. Defaults to 0.1.
        betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square. Defaults to (0.9, 0.999).
        t (float, optional): adaptivity parameter. Defaults to 0.001.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        t: float = 0.001,
    ) -> None:
        super().__init__(learning_rate, betas, t)
        self._m: Optional[Dict[str, npt.NDArray[np.float32]]] = None
        self._v: Optional[Dict[str, npt.NDArray[np.float32]]] = None
        self._params: dict = {}

    def step(
        self, pseudo_gradient: Dict[str, npt.NDArray[np.float32]]
    ) -> Dict[str, npt.NDArray[np.float32]]:

        """a step to optimize parameters of server model with pseudo gradient.

        Args:
            pseudo_gradient (Dict[str, npt.NDArray[np.float32]]): the pseudo gradient of server model

        Returns:
            Dict[str, npt.NDArray[np.float32]]: parameters of server model after step
        """

        if self._m is None:
            self._m = dict()
            for key, value in pseudo_gradient.items():
                self._m[key] = np.zeros_like(value)

        if self._v is None:
            self._v = dict()
            for key, value in pseudo_gradient.items():
                self._v[key] = np.zeros_like(value)

        """
        m_t = β_1m_t−1 + (1−β_1)∆_t
        v_t = β_2v_t−1 + (1−β_2)∆^2_t
        """
        for key, value in pseudo_gradient.items():
            self._m[key] = self._beta1 * self._m[key] + (1 - self._beta1) * value
            self._v[key] = self._beta2 * self._v[key] + (1 - self._beta2) * np.square(
                value
            )

        """x_t+1 = x_t + η mt / (√v_t+τ)"""
        new_params = dict()
        for key, value in self._params.items():
            new_params[key] = value.reshape((-1)) + self._lr * self._m[key] / (
                np.sqrt(self._v[key]) + self._adaptivity
            )
            self._params[key] = new_params[key].reshape(value.shape)

        return new_params
