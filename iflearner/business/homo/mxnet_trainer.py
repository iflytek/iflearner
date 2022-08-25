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
from typing import Dict

import mxnet as mx
import numpy as np
import numpy.typing as npt

from iflearner.business.homo.trainer import Trainer


class MxnetTrainer(Trainer):
    """implement the 'get' and 'set' function for the usual mxnet trainer."""

    def __init__(self, model: mx.gluon.nn.Sequential) -> None:
        self._model = model

    def get(
        self, param_type: Trainer.ParameterType = Trainer.ParameterType.ParameterModel
    ) -> Dict[str, npt.NDArray[np.float32]]:  # type: ignore
        """get parameters form the client, maybe the model parameter or
        gradient.

        Args:
            param_type: Param_type is ParameterModel or ParameterGradient, default is ParameterModel.

        Returns:
            dict, k: str (the parameter name), v: np.ndarray (the parameter value)
        """
        parameters = dict()
        for key, val in self._model.collect_params(".*weight").items():
            p = val.data().asnumpy()
            parameters[key] = p
        return parameters

    def set(
        self,
        parameters: Dict[str, npt.NDArray[np.float32]],  # type: ignore
        param_type: Trainer.ParameterType = Trainer.ParameterType.ParameterModel,
    ) -> None:
        """set parameters to the client, maybe the model parameter or gradient.

        Args:
            parameters: Parameters is the same as the return of 'get' function.
            param_type: Param_type is ParameterModel or ParameterGradient, default is ParameterModel.

        Returns: None
        """
        for key, value in parameters.items():
            self._model.collect_params().setattr(key, value)
