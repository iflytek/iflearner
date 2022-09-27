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
import json
from importlib import import_module
from threading import Thread
from typing import Any, Dict, Union

from iflearner.communication.mpc.piss import piss_server
from iflearner.communication.base import base_server
from iflearner.communication.mpc.piss import piss_pb2, piss_pb2_grpc
from iflearner.business.mpc.piss import piss_client_controller

class PissAggregateServer:

    def __init__(self, addr: str ,party_name: str) -> None:
        self._addr = addr
        self._party_name  = party_name
        self._piss_server_inst = piss_server.PissServer(party_name= party_name)

    def run(self) -> None:
        """start piss server"""
        base_server.start_server(self._addr,self._piss_server_inst)

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--addr", help="the server address", default="127.0.0.1:12095", type=str
    )
    parser.add_argument(
        "--name", help="the server name", default="server", type=str
    )
    args = parser.parse_args()

    global server 
    server = PissAggregateServer(addr = args.addr, party_name= args.name)
    server.run()

if __name__ == "__main__":
    main()

