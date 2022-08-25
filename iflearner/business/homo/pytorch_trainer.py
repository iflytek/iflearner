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

import numpy as np
import numpy.typing as npt
import torch

from iflearner.business.homo.trainer import Trainer


class PyTorchTrainer(Trainer):
    """implement the 'get' and 'set' function for the usual pytorch trainer."""

    def __init__(self, model: torch.nn.Module) -> None:
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
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                if param_type == self.ParameterType.ParameterModel:
                    parameters[name] = param.cpu().detach().numpy()
                else:
                    parameters[name] = param.grad.cpu().detach().numpy()

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
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                if param_type == self.ParameterType.ParameterModel:
                    param.data.copy_(torch.from_numpy(parameters[name]))
                else:
                    param.grad.copy_(torch.from_numpy(parameters[name]))
