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

from loguru import logger

from iflearner.business.homo.strategy import strategy_server
from iflearner.communication.homo import homo_pb2, message_type
from iflearner.communication.homo.homo_exception import HomoException


class FedavgServer(strategy_server.StrategyServer):
    """Implement the strategy of fedavg on server side.

    Attributes:
        num_clients (int): client number
        total_epoch (int): the epoch number of client trainning
        scaffold (bool): if use the scaffold method. Defaults to False.
        weighted_fedavg (bool): if use the weighted sum. Defaults to False.
    """

    def __init__(
        self,
        num_clients: int,
        total_epoch: int,
        scaffold: bool = False,
        weighted_fedavg: bool = False,
    ) -> None:
        super().__init__(num_clients, total_epoch)

        self._scaffold = scaffold
        self._weighted_fedavg = weighted_fedavg
        logger.info(
            f"num_clients: {self._num_clients}, scaffold: {self._scaffold}, weighted_fedavg: {self._weighted_fedavg}"
        )

        self._clients_samples: dict = {}

    def handler_register(
        self, party_name: str, sample_num: Optional[int] = 0, step_num: int = 0
    ) -> homo_pb2.RegistrationResponse:
        super().handler_register(party_name, sample_num, step_num)

        if self._weighted_fedavg:
            if sample_num == 0:
                raise ValueError(
                    "In weighted_fedavg mode, `sample_num` cannot be zero. You should implement `config` method in your `Trainer`"
                )
            self._clients_samples[party_name] = sample_num

        if self._scaffold:
            return homo_pb2.RegistrationResponse(
                strategy=message_type.STRATEGY_SCAFFOLD, parameters=None
            )

        return homo_pb2.RegistrationResponse(
            strategy=message_type.STRATEGY_FEDAVG, parameters=None
        )

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        logger.info(f"Client: {party_name}, epoch: {data.epoch}")

        if party_name not in self._training_clients:
            raise HomoException(
                HomoException.HomoResponseCode.Forbidden, "Client not notified."
            )

        if data.metrics is not None:
            for k, v in data.metrics.items():
                self._metric.add(k, party_name, data.epoch, v)

        self._training_clients[party_name]["param"] = data.parameters
        self._uploaded_num += 1
        if self._uploaded_num == self._num_clients:

            # calculate contributions of each client
            if self._weighted_fedavg:
                total_samples = 0
                for client_name in self._training_clients:
                    total_samples += self._clients_samples[client_name]

            self._uploaded_num = 0
            aggregate_result = dict()
            logger.info(f"Avg params, param num: {len(data.parameters)}")
            for param_name, param_info in data.parameters.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=param_info.shape
                )
                params = []

                for client_name, v in self._training_clients.items():
                    if self._weighted_fedavg:
                        params.append(
                            [
                                x * (self._clients_samples[client_name] / total_samples)
                                for x in v["param"][param_name].values
                            ]
                        )
                    else:
                        params.append(
                            [
                                x / self._num_clients
                                for x in v["param"][param_name].values
                            ]
                        )

                avg_param = [sum(x) for x in zip(*params)]
                if self._scaffold and self._server_param is not None:
                    # aggregate_result[param_name].values.extend(avg_param)
                    aggregate_result[param_name].values.extend(
                        [
                            sum(x)
                            for x in zip(
                                # type: ignore
                                avg_param,
                                self._server_param[param_name].values,
                            )
                        ]
                    )
                else:
                    aggregate_result[param_name].values.extend(avg_param)

            self._server_param = aggregate_result  # type: ignore
            self._on_aggregating = True
