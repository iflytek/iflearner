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
from email import message
from operator import imod
from tkinter import NO
from typing import Any
from loguru import logger
from iflearner.communication.base import base_pb2,base_server,constant


class PissBase(base_server.BaseServer):
    def __init__(self, party_name: str,route: str = None,cert_path: str = None) -> None:
        self._options = [
            ("grpc.max_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_send_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_receive_message_length", constant.MAX_MSG_LENGTH),
        ]
        self._party_name = party_name
        self._route = route
        self._cert_path = cert_path

    def _send(self, stub, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call send function."""

        resp = stub.send(req)
        return resp

    def _post(self, stub, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call post function."""

        resp = stub.post(req)
        return resp

    def _callback(self, stub, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call callback function."""

        resp = stub.callback(req)
        return resp