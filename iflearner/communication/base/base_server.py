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
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any

import grpc

from iflearner.communication.base import base_pb2, base_pb2_grpc, constant


class BaseServer(base_pb2_grpc.BaseServicer, ABC):
    """Provides methods that implement functionality of base server."""

    @abstractmethod
    def send(self, request: base_pb2.BaseRequest, context: Any) -> None:
        pass

    @abstractmethod
    def post(self, request: base_pb2.BaseRequest, context: Any) -> None:
        pass

    @abstractmethod
    def callback(self, request: base_pb2.BaseRequest, context: Any) -> None:
        pass


def start_server(addr: str, servicer: BaseServer) -> None:
    """Start server at the address."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_send_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_receive_message_length", constant.MAX_MSG_LENGTH),
        ],
    )
    base_pb2_grpc.add_BaseServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()
