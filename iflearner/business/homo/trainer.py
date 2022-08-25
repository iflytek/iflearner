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
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Dict

import numpy as np
import numpy.typing as npt


class Trainer(ABC):
    """The base class of trainer."""

    class ParameterType(IntEnum):
        """Define the type of parameter."""

        ParameterModel = auto()
        ParameterGradient = auto()

    @abstractmethod
    def get(
        self, param_type: ParameterType = ParameterType.ParameterModel
    ) -> Dict[str, npt.NDArray[np.float32]]:  # type: ignore
        """get parameters form the client, maybe the model parameter or
        gradient.

        Args:
            param_type: Param_type is ParameterModel or ParameterGradient, default is ParameterModel.

        Returns:
            dict, k: str (the parameter name), v: np.ndarray (the parameter value)
        """
        pass

    @abstractmethod
    def set(
        self,
        parameters: Dict[str, npt.NDArray[np.float32]],  # type: ignore
        param_type: ParameterType = ParameterType.ParameterModel,
    ) -> None:
        """set parameters to the client, maybe the model parameter or gradient.

        Args:
            parameters: Parameters is the same as the return of 'get' function.
            param_type: Param_type is ParameterModel or ParameterGradient, default is ParameterModel.

        Returns: None
        """
        pass

    # @abstractmethod
    def config(self) -> Dict[str, float]:
        """get training configuration.

        Returns:
            return a dict, at least including the following keys:
            learning_rate
            batch_num
            sample_num
        """
        return dict()

    @abstractmethod
    def fit(self, epoch: int) -> None:
        """fit model on one epoch.

        Args:
            epoch:  the current index of epoch

        Returns:
            None
        """
        pass

    @abstractmethod
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """evaluate model and return metrics.

        Args:
            epoch: the current index of epoch

        Returns:
            dict, k: str (metric name), v: float (metric value)
        """
        pass
