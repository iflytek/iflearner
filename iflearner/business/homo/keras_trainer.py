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

import keras
import numpy as np
import numpy.typing as npt

from iflearner.business.homo.trainer import Trainer


class KerasTrainer(Trainer):
    """implement the 'get' and 'set' function for the usual keras trainer."""

    def __init__(self, model: keras.models.Sequential) -> None:
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
        for item in self._model.layers:
            if item.name is not None:
                i = 0
                for weight in item.get_weights():
                    parameters[f"{item.name}-{i}"] = weight
                    i += 1
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
        for item in self._model.layers:
            if item.name is not None and len(item.get_weights()) > 0:
                i = 0
                weights = []
                while True:
                    i_name = f"{item.name}-{i}"
                    if i_name in parameters:
                        weights.append(parameters[i_name])
                    else:
                        break
                    i += 1
                item.set_weights(weights)
