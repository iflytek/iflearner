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
from iflearner.communication.mpc.piss import piss_client
from iflearner.communication.base import base_server
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss import piss_pb2
from iflearner.communication.base import base_server


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
        self._piss_client_inst = piss_client.PissClient(party_name= self._args.name,
                                                    server_addr = self._args.server,
                                                    cert_path=  self._args.cert,
                                                    data_path= self._args.data,
                                                    encryption_param = self._args.param
                                                    )
    def start_querty(self):
        self._piss_client_inst.start_querty()

    def get_secrets_sum(self):
        secrets_sum = self._piss_client_inst.get_secrets_sum()
        return secrets_sum
