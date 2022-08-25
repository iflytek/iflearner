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
from abc import ABC
from enum import IntEnum, auto
from typing import Any, Dict

import numpy as np

from iflearner.communication.homo import homo_pb2


class StrategyClient(ABC):
    """Implement the strategy of client.

    Attributes:
        _custom_handlers (Dict[str, Any]): custom handlers
        _trainer_config (Dict[str, Any]): the trainer config of client
        _current_stage (Stage): current stage of client
        _aggregate_result: aggregate model parameters result with grpc format
        _aggregate_result_np: aggregate model parameters result with numpy format
        _aggregate_c: None
        _local_c = None
        _local_c_initial = None
        _smpc (bool): use smpc to start federated learning if _smpc is True
        _sum_random_value: random value using in smpc
    """

    class Stage(IntEnum):
        """Enum the stage of client."""

        Waiting = auto()
        Training = auto()
        Setting = auto()

    def __init__(self) -> None:
        self._custom_handlers: Dict[str, Any] = dict()
        self._trainer_config: Dict[str, Any] = dict()
        self._current_stage = self.Stage.Waiting
        self._aggregate_result = None
        self._aggregate_result_np = None
        self._aggregate_c = None
        self._local_c = None
        self._local_c_initial = None
        self._smpc = False
        self._sum_random_value = 0.0
        self._gradient_suffix = "_gradient"

    @property
    def custom_handlers(self) -> Dict[str, Any]:
        return self._custom_handlers

    def set_trainer_config(self, config: Dict[str, Any]) -> None:
        """set trainer config.

        Args:
            config (Dict[str, Any]): the config of client Trainer
        """
        self._trainer_config = config

    def generate_registration_info(self) -> None:
        """Generate the message of MSG_REGISTER."""
        pass

    def aggregate_result(self) -> homo_pb2.AggregateResult:
        """get the aggregated model parameters.

        Returns:
            homo_pb2.AggregateResult: the aggregated model parameters of grpc format
        """
        return self._aggregate_result_np

    def generate_upload_param(
        self,
        epoch: int,
        data: Dict[Any, Any],
        metrics: Dict[str, float] = None,
    ) -> Any:
        """Generate the message of MSG_UPLOAD_PARAM.

        Args:
            epoch (int): Current epoch of number of client training.
            data (Dict[Any, Any]): The data that will be uploaded to server.
            metrics (Dict[str, Any]): The client metrics.

        Returns:
            Any: The grpc format data that can be send to server
        """
        pb_params = dict()
        for k, v in data.items():
            pb_params[k] = homo_pb2.Parameter(values=v.ravel(), shape=v.shape)

        data = homo_pb2.UploadParam(epoch=epoch, parameters=pb_params, metrics=metrics)

        return data

    def update_param(self, data: homo_pb2.AggregateResult) -> homo_pb2.AggregateResult:
        """Update the parameter during training.

        Args:
            data (homo_pb2.AggregateResult): the aggregated result from server

        Returns:
            homo_pb2.AggregateResult: the updated result
        """
        pass

    def handler_aggregate_result(self, data: homo_pb2.AggregateResult) -> None:
        """Handle the message of MSG_AGGREGATE_RESULT from the server.

        Args:
            data (homo_pb2.AggregateResult): the aggregated result from server
        """

        data_m = dict()
        data_c = dict()
        for k, v in data.parameters.items():
            if k.endswith(self._gradient_suffix):
                data_c[k.replace(self._gradient_suffix, "")] = homo_pb2.Parameter(
                    shape=v.shape
                )
                data_c[k.replace(self._gradient_suffix, "")].values.extend(v.values)
            else:
                data_m[k] = homo_pb2.Parameter(shape=v.shape)
                data_m[k].values.extend(v.values)

        self._aggregate_result = homo_pb2.AggregateResult(parameters=data_m)
        self._aggregate_result_np = {}  # type: ignore
        for k, v in data_m.items():
            self._aggregate_result_np[k] = np.asarray(v.values).reshape(v.shape)  # type: ignore

        self._aggregate_c = homo_pb2.AggregateResult(parameters=data_c)
        self._current_stage = self.Stage.Setting

    def handler_notify_training(self) -> None:
        """Handle the message of MSG_NOTIFY_TRAINING from the server."""
        self._current_stage = self.Stage.Training

    def current_stage(self) -> Stage:
        """the current stage, which is in Waiting, Trainning or Settinh stage.

        Returns:
            Stage: the current stage
        """
        return self._current_stage

    def set_current_stage(self, stage: Stage) -> None:
        """set current stage."""

        self._current_stage = stage

    def set_global_param(self, param: Dict[str, Any]) -> None:
        """set global parameters.

        Args:
            param (Dict[str, Any]): parameters
        """

        self._global_param = param
