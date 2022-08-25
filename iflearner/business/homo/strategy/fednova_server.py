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
from typing import Optional

import numpy as np
from loguru import logger

from iflearner.business.homo.strategy import strategy_server
from iflearner.communication.homo import homo_pb2, message_type


class FedNovaServer(strategy_server.StrategyServer):
    """Implement the strategy of fednova on server side.

    Attributes:
        num_clients (int): client number
        total_epoch (int): the epoch number of client trainning
        _clients_samples (dict): samples of each client
        _step_nums (dict): step numbers for each client to optimize its model
    """

    def __init__(
        self,
        num_clients: int,
        total_epoch: int,
    ) -> None:
        super().__init__(num_clients, total_epoch)
        logger.info(f"num_clients: {self._num_clients}")

        self._clients_samples: dict = {}
        self._step_nums: dict = {}

    def handler_register(
        self,
        party_name: str,
        sample_num: Optional[int] = 0,
        step_num: Optional[int] = 0,
    ) -> homo_pb2.RegistrationResponse:
        super().handler_register(party_name)

        if sample_num == 0 or step_num == 0:
            raise Exception(
                "`sample_num` and `step_num` cannot be zero. You should implement `config` method in your `Trainer`"
            )
        self._clients_samples[party_name] = sample_num

        self._step_nums[party_name] = step_num

        return homo_pb2.RegistrationResponse(strategy=message_type.STRATEGY_FEDNOVA)

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        super().handler_upload_param(party_name, data)

        if self._uploaded_num == self._num_clients:
            # calculate the total samples and tau_eff of all clients
            total_samples = 0
            tau_eff = 0.0
            for client_name in self._training_clients:
                total_samples += self._clients_samples[client_name]
                tau_eff += (
                    self._step_nums[client_name]
                    * self._clients_samples[client_name]
                    / total_samples
                )

            self._uploaded_num = 0
            aggregate_result = dict()
            avg_update = {
                name: param.reshape(-1) for name, param in self._params.items()
            }
            logger.info(f"Aggregate params, param num: {len(data.parameters)}")

            for param_name, param in self._params.items():
                for client_name, v in self._training_clients.items():
                    d = (
                        np.array(v["param"][param_name].values)
                        / self._step_nums[client_name]
                        * self._clients_samples[client_name]
                        / total_samples
                        * tau_eff
                    )
                    avg_update[param_name] += d

            for param_name, param in avg_update.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=self._params[param_name].shape
                )
                aggregate_result[param_name].values.extend(param.tolist())

            self._server_param = aggregate_result  # type: ignore
            self._on_aggregating = True
