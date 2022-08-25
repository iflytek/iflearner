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
from typing import List

from loguru import logger

from iflearner.communication.base import base_client, base_pb2
from iflearner.communication.peer import aes, message_type
from iflearner.communication.peer.diffie_hellman_inst import DiffieHellmanInst


class PeerClient(base_client.BaseClient):
    """The client for peer party communication.
    """

    def __init__(self, server_addr: str, party_name: str, peer_cert: str = None) -> None:
        super().__init__(server_addr, peer_cert)
        self._party_name = party_name

    def get_DH_public_key(self) -> List:
        """Get Diffie-Hellman public key from specified server.

        Returns:
            The public key.
        """

        while True:
            try:
                req = base_pb2.BaseRequest(
                    party_name=self._party_name,
                    type=message_type.MSG_DH_PUBLIC_KEY,
                    data=DiffieHellmanInst.generate_public_key(),
                )
                resp = self._send(req)
                public_key = resp.data

                logger.info(f"Public key: {public_key}")
                return public_key
            except Exception as e:
                logger.info(e)
                time.sleep(3)

    def get_SMPC_random_key(self, key: str) -> float:
        """Get random value from the other party.

        Returns:
            A float value.
        """

        req = base_pb2.BaseRequest(
            party_name=self._party_name, type=message_type.MSG_SMPC_RANDOM_KEY
        )
        resp = self._send(req)

        random_float = float(aes.AESCipher(key).decrypt(resp.data))
        logger.info(f"Random float: {random_float}")

        return random_float
