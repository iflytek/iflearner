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
import json
import os
import time
from abc import ABC
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from iflearner.communication.homo import homo_pb2, message_type
from iflearner.communication.homo.homo_exception import HomoException
from iflearner.business.util.metric import Metric


class ClientStatus:
    def __init__(self, total_epoch: int) -> None:
        self.ready = False
        self.complete = False
        self.current_epoch = 0
        self.total_epoch = total_epoch


class StrategyServer(ABC):

    """Implement the strategy of server.

    Args:
        num_clients (int): client number
        total_epoch (int): the epoch number of client trainning

    Attributes:
        _num_clients (int) : client number
        _total_epoch (int): the epoch number of client trainning
        _custom_handlers (Dict[str, Any]):
        _complete_num (int): the number of client that has completed federated learning
        _clients (Dict[str, ClientStatus]) : a dict storage the client status
        _training_clients (dict) : a dict storage the training clients
        _server_param (dict): the server model
        _ready_num (int): the number of ready client
        _uploaded_num (int): the number of client that has uploaded its model parameters
        _aggregated_num (int): the number of client that has aggregated its model parameters
        _on_aggregating (bool): whether the server is in aggregating stage
        _params (dict): the server model parameters
    """

    def __init__(self, num_clients, total_epoch) -> None:
        self._num_clients = num_clients
        self._total_epoch = total_epoch
        self._custom_handlers: Dict[str, Any] = dict()
        self._complete_num: int = 0
        self._clients: Dict[str, ClientStatus] = {}
        self._training_clients: dict = {}
        self._server_param = None
        self._ready_num = 0
        self._uploaded_num = 0
        self._aggregated_num = 0
        self._on_aggregating = False
        self._params: dict = {}
        self._metric = Metric(logdir="metric")

    def _exit(self) -> None:
        """exit the federated learning."""

        def sleep():
            time.sleep(3)
            os._exit(0)

        Thread(target=sleep).start()

    def clients_to_json(self) -> str:
        """save clients to json file.

        Returns:
            str: json string
        """

        tmp = dict()
        for k, v in self._clients.items():
            print(k, v)
            tmp[k] = v.__dict__

        return json.dumps(tmp)

    def handler_complete(self, party_name: str) -> None:
        """Handle the message of MSG_COMPLETE from the client.

        Args:
            party_name (str): client name

        Raises:
            HomoException: if party_name not in the register list, raise the Unauthorized error
        """
        logger.info(f"Client complete: {party_name}")
        if party_name not in self._clients:
            raise HomoException(
                HomoException.HomoResponseCode.Unauthorized, "Unregistered client."
            )

        self._complete_num += 1
        self._clients[party_name].complete = True

        # if self._complete_num == self._num_clients:
        #     self._exit()

    @property
    def custom_handlers(self) -> Dict[str, Any]:
        return self._custom_handlers

    def handler_register(
        self, party_name: str, sample_num: Optional[int] = 0, step_num: int = 0
    ) -> homo_pb2.RegistrationResponse:
        """Handle the message of MSG_REGISTER from the client.

        Args:
            party_name (str): client name
            sample_num (Optional[int], optional): the total sample number of client `party_name` . Defaults to 0.
            step_num (int, optional): The number a client epoch needs to be optimized, always equals to the batch number of client. Defaults to 0.

        Raises:
            HomoException: _description_

        Returns:
            homo_pb2.RegistrationResponse: if party_name not in the register list, raise the Unauthorized error
        """
        logger.info(f"Client register: {party_name}")
        if len(self._clients) >= self._num_clients:
            raise HomoException(
                HomoException.HomoResponseCode.Unauthorized,
                "Registered clients are full.",
            )

        self._clients[party_name] = ClientStatus(self._total_epoch)

    def handler_client_ready(self, party_name: str) -> None:
        """Handle the message of MSG_CLIENT_READY from the client."""
        logger.info(f"Client ready: {party_name}")
        if party_name not in self._clients:
            raise HomoException(
                HomoException.HomoResponseCode.Unauthorized, "Unregistered client."
            )

        self._clients[party_name].ready = True
        self._clients[party_name].current_epoch += 1
        self._ready_num += 1
        if self._ready_num == self._num_clients:
            logger.info("Clients are all ready.")
            self._ready_num = 0
            for k in self._clients.keys():
                self._training_clients[k] = dict()

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        """Handle the message of MSG_UPLOAD_PARAM from the client.

        Args:
            party_name (str): client name
            data (homo_pb2.UploadParam): the data uploaded from `party_name`, with grpc format

        Raises:
            HomoException:  if party_name not in the training_clients list, raise the Forbidden error
        """
        logger.info(f"Client: {party_name}, epoch: {data.epoch}")
        if party_name not in self._training_clients:
            raise HomoException(
                HomoException.HomoResponseCode.Forbidden, "Client not notified."
            )

        self._training_clients[party_name]["param"] = data.parameters
        self._uploaded_num += 1
        if self._params is None:
            self._params = dict()
            for param_name, param_info in data.parameters.items():
                self._params[param_name] = np.array(param_info.values).reshape(
                    param_info.shape
                )

        if data.metrics is not None:
            for k, v in data.metrics.items():
                self._metric.add(k, party_name, data.epoch, v)

    def get_client_notification(self, party_name: str) -> Tuple[str, Any]:
        """Get the notification information of the specified client.

        Args:
            party_name (str): client name

        Returns:
            Tuple[str, Any]: the notification message type and notification data
        """
        if party_name in self._training_clients:
            if self._on_aggregating:
                if not self._training_clients[party_name].get("aggregating", False):
                    self._training_clients[party_name]["aggregating"] = True
                    result = homo_pb2.AggregateResult(parameters=self._server_param)

                    self._aggregated_num += 1
                    if self._aggregated_num == self._num_clients:
                        self._aggregated_num = 0
                        self._on_aggregating = False
                        self._training_clients.clear()

                    return message_type.MSG_AGGREGATE_RESULT, result
            elif not self._training_clients[party_name].get("training", False):
                self._training_clients[party_name]["training"] = True
                return message_type.MSG_NOTIFY_TRAINING, None
        return "", None
