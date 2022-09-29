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
from abc import ABC
from typing import Any, Dict

from importlib import import_module
from typing import Any, Dict, Union
from loguru import logger
import grpc
from iflearner.communication.mpc import piss
from iflearner.communication.base import base_pb2_grpc


class PissStrategyBase(ABC):
    def __init__(self ,cert_path: str, party_name: str,options) -> None:

        self._cert_path = cert_path
        self._party_name = party_name
        self._options = options

        self._routes: dict = dict()
        self._stubs:  dict = dict()
        self._party_name_list = []

        self._initiator_party_name: str = str()
        self._initiator_route: str = str()
        self._initiator_stub = None 

    def generate_stub(self, destination_addr: str):

        if self._cert_path is None:
            channel = grpc.insecure_channel(destination_addr, options = self._options)
        else:
            with open(self._cert_path, "rb") as f:
                cert_bytes = f.read()

            channel = grpc.secure_channel(
                destination_addr, grpc.ssl_channel_credentials(cert_bytes), options = self._options
            )
        stub = base_pb2_grpc.BaseStub(channel)
        return stub 
