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
import argparse
from importlib import import_module
from typing import Any, Dict, Union
from loguru import logger

from iflearner.communication.mpc.piss import piss_client_services
from iflearner.communication.base import base_server
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss import piss_pb2
import grpc
from iflearner.communication.base import base_pb2, base_pb2_grpc,base_server,constant


class PissClientServicesController:

    def __init__(self, args: argparse.Namespace) -> None:

        self._args = args
        self._piss_client_services_inst = piss_client_services.PissClientServices(
                                                        server_addr=self._args.server ,
                                                        party_name= self._args.name,
                                                        route= self._args.addr,
                                                        data_path= self._args.data,
                                                        cert_path=  self._args.cert
                                                        )

    def run(self) -> None:
        # """REGISTER"""
        resp = self._piss_client_services_inst.transport(
                    type = message_type.MSG_REGISTER,
                    data = piss_pb2.RegistrationInfo(route = self._args.addr)
                )
        """start piss client server"""
        if resp.code == 0:
            base_server.start_server(self._args.addr,self._piss_client_services_inst)

class PissClientController:
    def __init__(self, args: argparse.Namespace) -> None:
        
        self._args = args
        self._options = [
                ("grpc.max_message_length", constant.MAX_MSG_LENGTH),
                ("grpc.max_send_message_length", constant.MAX_MSG_LENGTH),
                ("grpc.max_receive_message_length", constant.MAX_MSG_LENGTH),
            ]

        self._channel = grpc.insecure_channel(self._args.addr, options = self._options)
        self._stub = base_pb2_grpc.BaseStub(self._channel)
        self._encryption_param = self._args.param
        self._data_path = self._args.data
        self._party_name = self._args.name

    def init_data(self):

        data = piss_pb2.InitData(data_path = self._data_path)
        req = base_pb2.BaseRequest(party_name = self._party_name,
                               type = message_type.MSG_INIT_DATA,
                               data = data.SerializeToString())
        resp = self._stub.send(req)

    def start_querty(self):
        
        #encryption_param = {'10001':'Age', '10002':'Money'}
        #encryption_param = json.loads(self._encryption_param)
        data = piss_pb2.ShareEncryptionParam(encryption_param = self._encryption_param)

        req = base_pb2.BaseRequest(party_name = self._party_name,
                               type = message_type.MSG_START_QUERY,
                               data = data.SerializeToString())

        resp = self._stub.send(req)



