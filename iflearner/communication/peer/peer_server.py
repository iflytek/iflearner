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
import random
import time
from typing import Any

from loguru import logger

from iflearner.communication.base import base_pb2, base_server
from iflearner.communication.peer import aes, message_type
from iflearner.communication.peer.diffie_hellman_inst import DiffieHellmanInst


class PeerServer(base_server.BaseServer):
    """The server for peer party communication."""

    def __init__(self, peer_num: int) -> None:
        super().__init__()

        self._parties_secret = dict()
        self._parties_random_value = dict()
        self._peer_num = peer_num

    def sum_parties_random_value(self) -> float:
        """When the values from each party are received, we add all the values together."""

        while True:
            if self._peer_num == len(self._parties_random_value.values()):
                return sum(self._parties_random_value.values())

            time.sleep(1)

    def send(
        self, request: base_pb2.BaseRequest, context: Any
    ) -> base_pb2.BaseResponse:
        """Send two types of requests to server, including MSG_DH_PUBLIC_KEY and MSG_SMPC_RANDOM_KEY."""

        logger.info(f"IN: party: {request.party_name}, message type: {request.type}")

        if request.type == message_type.MSG_DH_PUBLIC_KEY:
            self._parties_secret[
                request.party_name
            ] = DiffieHellmanInst().generate_secret(request.data)

            return base_pb2.BaseResponse(data=DiffieHellmanInst().generate_public_key())
        elif request.type == message_type.MSG_SMPC_RANDOM_KEY:
            random_float = random.uniform(0.1, 1.0)
            logger.info(f"Party: {request.party_name}, Random float: {random_float}")
            self._parties_random_value[request.party_name] = -random_float

            # return base_pb2.BaseResponse(data=bytearray(struct.pack('f', random_float)))
            return base_pb2.BaseResponse(
                data=aes.AESCipher(self._parties_secret[request.party_name]).encrypt(
                    str(random_float)
                )
            )

    def post(self, request: Any, context: Any) -> None:
        pass

    def callback(self, request: Any, context: Any) -> None:
        pass
