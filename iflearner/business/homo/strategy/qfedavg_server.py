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
from typing import Dict, Optional

import numpy as np
from loguru import logger

from iflearner.business.homo.strategy import strategy_server
from iflearner.communication.homo import homo_pb2, message_type


class qFedavgServer(strategy_server.StrategyServer):
    """Implement the strategy of qFedavg on server side.

    Attributes:
        num_clients (int): client number
        total_epoch (int): the epoch number of client trainning
        q (float): q factor. Defaults to 1.
        learning_rate (float): learning rate. Defaults to 0.1.
        _fs (dict): loss values of each client
    """

    def __init__(
        self,
        num_clients: int,
        total_epoch: int,
        q: float = 1,
        learning_rate: float = 0.1,
    ) -> None:
        super().__init__(num_clients, total_epoch)
        logger.info(f"num_clients: {self._num_clients}, strategy: qFedavg")

        self._q = q
        self._lr = learning_rate
        self._params: dict = {}
        self._fs: dict = {}

    def handler_register(
        self, party_name: str, sample_num: Optional[int] = None, step_num: int = 0
    ) -> homo_pb2.RegistrationResponse:
        super().handler_register(party_name)

        return homo_pb2.RegistrationResponse(strategy=message_type.STRATEGY_qFEDAVG)

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        if "loss" not in data.parameters:
            raise Exception(
                "In qFedavg strategy, you shuole add `loss` key in the return value of Trainer.get() method"
            )
        else:
            self._fs[party_name] = data.parameters.pop("loss").values[0]
        super().handler_upload_param(party_name, data)

        if self._uploaded_num == self._num_clients:
            self._uploaded_num = 0
            aggregate_result = dict()

            logger.info(f"Fadopt params, param num: {len(data.parameters)}")

            grads = dict()
            deltas = dict()
            hs = dict()
            for client_name, v in self._training_clients.items():
                grad = dict()
                delta = dict()
                for name, param in self._params.items():
                    grad[name] = (
                        (param.reshape((-1)) - np.array(v["param"][name].values))
                        * 1.0
                        / self._lr
                    )
                    delta[name] = (
                        np.float_power(self._fs[client_name] + 1e-10, self._q)
                        * grad[name]
                    )
                grads[client_name] = grad
                deltas[client_name] = delta
                hs[client_name] = self._q * np.float_power(
                    self._fs[client_name] + 1e-10, (self._q - 1)
                ) * self.norm_grad(grads[client_name]) + (
                    1.0 / self._lr
                ) * np.float_power(
                    self._fs[client_name] + 1e-10, self._q
                )

            new_param = self.step(deltas, hs)

            for param_name, param in new_param.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=self._params[param_name].shape
                )
                aggregate_result[param_name].values.extend(param.tolist())

            self._server_param = aggregate_result  # type: ignore
            self._on_aggregating = True

    def norm_grad(self, grad: Dict[str, Dict]):
        """normalize the grad.

        Args:
            grad (Dict[str, Dict]): grad

        Returns:
            _type_: the normalized grad
        """
        sum_grad = 0
        for v in grad.values():
            sum_grad += np.sum(np.square(v))  # type: ignore
        return sum_grad

    def step(self, deltas: Dict[str, Dict], hs: Dict[str, float]):
        """a optimized step for deltas.

        Args:
            deltas (Dict[str, Dict]): the delta of model parameters
            hs (Dict[str, float]): demominator

        Returns:
            _type_: new parameters after optimizing the deltas
        """
        demominator = sum(hs.values())
        updates: dict = {}
        for client_delta in deltas.values():
            for param_name, param in client_delta.items():
                updates[param_name] = updates.get(param_name, 0) + param / demominator
        new_param = {}
        for param_name, param in self._params.items():
            new_param[param_name] = param.reshape((-1)) - updates[param_name]
            self._params[param_name] = new_param[param_name].reshape(param.shape)
        return new_param
