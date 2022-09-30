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

from operator import imod
import timeit
from tkinter import NO
from typing import Any

from loguru import logger
import time

from iflearner.communication.base import base_pb2
from iflearner.communication.mpc.piss import piss_pb2
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss.piss_exception import PissException
import grpc
from iflearner.communication.base import base_pb2, base_pb2_grpc,base_server,constant

class PissClient():
    def __init__(self, party_name:str, server_addr:str, cert_path:str, data_path, encryption_param) -> None:

        self._options = [
            ("grpc.max_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_send_message_length", constant.MAX_MSG_LENGTH),
            ("grpc.max_receive_message_length", constant.MAX_MSG_LENGTH),
        ]
        self._party_name = party_name
        self._cert_path = cert_path
        self._server_addr = server_addr
        if self._cert_path is None:
            channel = grpc.insecure_channel(self._server_addr, options = self._options)
        else:
            with open(self._cert_path, "rb") as f:
                cert_bytes = f.read()

            channel = grpc.secure_channel(
                self._server_addr, grpc.ssl_channel_credentials(cert_bytes), options = self._options
            )
        self._stub = base_pb2_grpc.BaseStub(channel)
        self._encryption_param = encryption_param
        self._data_path = data_path
        self._secrets_sum = None

    def transport(self, type: str, data: Any = None)-> None:
        """Transport data betwees client and server or client."""
        try:
            start = timeit.default_timer()
            req = base_pb2.BaseRequest(party_name = self._party_name,
                                        type = type)
            if data is not None:
                req.data = data.SerializeToString()
            resp = None 
            if type == message_type.MSG_INIT_DATA:
                resp = self._stub.send(req)
            elif type == message_type.MSG_START_QUERY:
                resp = self._stub.send(req)
            elif type == message_type.MSG_GET_SUM_SECRETS:
                resp = self._stub.callback(req)
            elif type == message_type.MSG_END_QUERY:
                resp = self._stub.callback(req)
            if resp.code != 0:  # type: ignore
                raise PissException(code=PissException.PissResponseCode(resp.code), message=resp.message  # type: ignore
                    )
        except PissException as e:
            logger.info(str(e))
        finally:
            stop = timeit.default_timer()
            logger.info(f"OUT: message type: {type}, time: {1000 * (stop - start)}ms")
        return resp

    def init_data(self):
        data = piss_pb2.InitData(data_path = self._data_path)
        self.transport(type= message_type.MSG_INIT_DATA, data = data)

    def start_querty(self):
        data = piss_pb2.ShareEncryptionParam(encryption_param = self._encryption_param)
        self.transport(type= message_type.MSG_START_QUERY, data = data)

        while True:
            data = piss_pb2.CallBack(call_back_msg = message_type.MSG_GET_SUM_SECRETS)
            resp = self.transport(type= message_type.MSG_GET_SUM_SECRETS , data = data)
            if len(resp.data) != 0 :
                resp_data = piss_pb2.SecretsSUM()
                resp_data.ParseFromString(resp.data)
                self._secrets_sum = resp_data.secrets_sum
                data = piss_pb2.CallBack(call_back_msg = message_type.MSG_END_QUERY)
                self.transport(type= message_type.MSG_END_QUERY , data = data)
                break
            time.sleep(message_type.MSG_HEARTBEAT_INTERVAL)
    
    def get_secrets_sum(self):
        return self._secrets_sum
        



