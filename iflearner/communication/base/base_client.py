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
import grpc

from iflearner.communication.base import base_pb2, base_pb2_grpc, constant


class BaseClient:
    """Provides methods that implement functionality of base client."""

    def __init__(self, addr: str, cert_path: str = None) -> None:
        options = [
            ("grpc.max_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_send_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_receive_message_length", constant.MAX_MSG_LENGTH),
        ]

        if cert_path is None:
            channel = grpc.insecure_channel(addr, options=options)
        else:
            with open(cert_path, "rb") as f:
                cert_bytes = f.read()

            channel = grpc.secure_channel(
                addr, grpc.ssl_channel_credentials(cert_bytes), options=options
            )

        self._stub: base_pb2_grpc.BaseStub = base_pb2_grpc.BaseStub(channel)

    def _send(self, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call send function."""

        resp = self._stub.send(req)
        return resp

    def _post(self, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call post function."""

        resp = self._stub.post(req)
        return resp

    def _callback(self, req: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call callback function."""

        resp = self._stub.callback(req)
        return resp
