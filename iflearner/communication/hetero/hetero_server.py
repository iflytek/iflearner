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
from typing import Any, Dict, Tuple, List
from iflearner.communication.base import base_pb2, base_server


class HeteroServer(base_server.BaseServer):
    """Implement the server which saves the client requests.

    Attributes:
        messages (Dict[str, List[Tuple[str, bytes]]]): Save the client requests.
    """

    def __init__(self, party_with_role: Dict[str, str]) -> None:
        self.messages: Dict[str, List[Tuple[str, bytes]]] = dict()
        self._party_with_role = party_with_role

    def send(self, request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        pass

    def callback(self, request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        pass

    def post(self, request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        """Save the received data according to the party name and type.

        Args:
            request (base_pb2.BaseRequest): Request details.
            context (Any): No use.

        Returns:
            base_pb2.BaseResponse: Empty response.
        """
        role = self._party_with_role[request.party_name]
        logger.info(
            f"Receive message, role: {role}, party: {request.party_name}, step: {request.type}, data length: {len(request.data)}")
        if f"{role}.{request.type}" in self.messages:
            self.messages[f"{role}.{request.type}"].append(
                (request.party_name, request.data))
        else:
            self.messages[f"{role}.{request.type}"] = [
                (request.party_name, request.data)]
        return base_pb2.BaseResponse()
