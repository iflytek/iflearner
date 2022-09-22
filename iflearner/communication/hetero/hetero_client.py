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

from loguru import logger
from iflearner.communication.base import base_client, base_pb2


class HeteroClient(base_client.BaseClient):
    def __init__(self, addr: str, party_name: str, cert_path: str = None) -> None:
        super().__init__(addr, cert_path)
        self._party_name = party_name

    def post(self, type: str, data: bytes):
        """Post a specific type of data to the server.

        Args:
            type (str): Sign the data with type.
            data (bytes): The binary data.
        """
        req = base_pb2.BaseRequest(
            party_name=self._party_name, type=type, data=data)
        self._post(req)
