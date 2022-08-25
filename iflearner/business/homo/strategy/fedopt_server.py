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
from iflearner.business.homo.strategy.opt.fedopt import FedOpt
from iflearner.communication.homo import homo_pb2, message_type


class FedoptServer(strategy_server.StrategyServer):
    """Implement the strategy of fedopt on server side.

    Attributes:
        num_clients (int): client number
        total_epoch (int): the epoch number of client trainning
        opt (FedOpt): the FedOpt method, which is in FedAdam, FedAdagrad, FedYogi or FedAvgM
    """

    def __init__(
        self,
        num_clients: int,
        total_epoch: int,
        opt: FedOpt,
    ) -> None:
        super().__init__(num_clients, total_epoch)

        self._opt = opt

        logger.info(f"num_clients: {self._num_clients}, opt: {type(opt).__name__}")

    def handler_register(
        self, party_name: str, sample_num: Optional[int] = None, step_num: int = 0
    ) -> homo_pb2.RegistrationResponse:
        super().handler_register(party_name)

        return homo_pb2.RegistrationResponse(strategy=message_type.STRATEGY_FEDOPT)

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        super().handler_upload_param(party_name, data)

        if self._opt._params is None:
            self._opt.set_params(self._params)

        if self._uploaded_num == self._num_clients:
            self._uploaded_num = 0
            aggregate_result = dict()
            grad = dict()
            logger.info(f"Fadopt params, param num: {len(data.parameters)}")

            """delta T = avg(new_weight - old_weight) = avg(new_weight - gloabel_weight) = avg(new_weight) - gloabal"""

            for param_name, param_info in data.parameters.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=param_info.shape
                )
                params = []
                for v in self._training_clients.values():
                    params.append(v["param"][param_name].values)

                avg_param = [sum(x) / self._num_clients for x in zip(*params)]
                grad[param_name] = np.array(
                    avg_param, dtype="float32"
                ) - self._opt._params[param_name].reshape((-1))

            # to optimize server model using grad and opt
            new_param = self._opt.step(grad)
            for param_name, param in new_param.items():
                aggregate_result[param_name].values.extend(param.tolist())

            self._server_param = aggregate_result  # type: ignore
            self._on_aggregating = True
