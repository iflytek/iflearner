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
import time
import timeit
from typing import Any

from loguru import logger

from iflearner.business.homo.strategy import strategy_client
from iflearner.communication.base import base_client, base_pb2
from iflearner.communication.homo import homo_pb2, message_type
from iflearner.communication.homo.homo_exception import HomoException


class HomoClient(base_client.BaseClient):
    """Implement homogeneous client base on base_client.BaseClient."""

    def __init__(
        self, server_addr: str, party_name: str, cert_path: str = None
    ) -> None:
        super().__init__(server_addr, cert_path)
        self._party_name = party_name
        self._strategy: strategy_client.StrategyClient = None  # type: ignore

    def set_strategy(self, strategy: strategy_client.StrategyClient) -> None:
        self._strategy = strategy

    def transport(self, type: str, data: Any = None) -> homo_pb2.RegistrationResponse:
        """Transport data to server."""
        start = timeit.default_timer()

        req = base_pb2.BaseRequest(party_name=self._party_name, type=type)
        if data is not None:
            req.data = data.SerializeToString()

        resp = None
        if type == message_type.MSG_REGISTER:
            resp = self._send(req)
        elif type == message_type.MSG_CLIENT_READY:
            resp = self._send(req)
        elif type == message_type.MSG_UPLOAD_PARAM:
            resp = self._post(req)
        elif type == message_type.MSG_COMPLETE:
            resp = self._send(req)

        stop = timeit.default_timer()
        logger.info(f"OUT: message type: {type}, time: {1000 * (stop - start)}ms")

        if resp.code != 0:  # type: ignore
            raise HomoException(
                code=HomoException.HomoResponseCode(resp.code), message=resp.message  # type: ignore
            )

        if type == message_type.MSG_REGISTER:
            data = homo_pb2.RegistrationResponse()
            data.ParseFromString(resp.data)  # type: ignore
            return data

    def notice(self) -> None:
        """Receive notifications from the server regularly."""
        while True:
            start = timeit.default_timer()

            req = base_pb2.BaseRequest(party_name=self._party_name)
            resp = self._callback(req)
            if resp.code != 0:
                raise HomoException(
                    code=HomoException.HomoResponseCode(resp.code), message=resp.message
                )

            if resp.type == message_type.MSG_AGGREGATE_RESULT:
                data = homo_pb2.AggregateResult()
                data.ParseFromString(resp.data)
                self._strategy.handler_aggregate_result(data)
            elif resp.type == message_type.MSG_NOTIFY_TRAINING:
                self._strategy.handler_notify_training()
            elif resp.type in self._strategy.custom_handlers:
                self._strategy.custom_handlers[resp.type]()  # type: ignore

            if resp.type != "":
                stop = timeit.default_timer()
                logger.info(
                    f"IN: party: message type: {resp.type}, time: {1000 * (stop - start)}ms"
                )

            time.sleep(message_type.MSG_HEARTBEAT_INTERVAL)
