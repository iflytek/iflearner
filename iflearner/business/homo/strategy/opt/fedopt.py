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
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt


class FedOpt:
    """Implementation based on https://arxiv.org/abs/2003.00295.

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
        self._lr = learning_rate
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._adaptivity = t
        self._params: dict = {}

    def step(
        self,
        pseudo_gradient: Dict[str, npt.NDArray[np.float32]],
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """a step to optimize parameters of server model with pseudo gradient.

        Args:
            pseudo_gradient (Dict[str, npt.NDArray[np.float32]]): the pseudo gradient of server model

        Returns:
            Dict[str, npt.NDArray[np.float32]]: parameters of server model after step
        """
        pass

    def set_params(self, params):
        """set params to self._params.

        Args:
            params (_type_): parameters of server model
        """
        self._params = params
