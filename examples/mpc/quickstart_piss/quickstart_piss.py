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
from loguru import logger
import time

from iflearner.communication.mpc.piss import piss_client_services
from iflearner.business.mpc.piss import piss_strategy_client
from iflearner.communication.base import base_server
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss import piss_pb2, piss_pb2_grpc
from iflearner.business.mpc.piss.piss_client_controller import PissClientController
from iflearner.business.mpc.piss.argument import parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default='examples/mpc/quickstart_piss/piss_data_test.csv',
        type=str,
        help="path of data")

    parser.add_argument(
        "--param",
        default={'10001':'Age', '10002':'Money'},
        type=json.loads,
        help="encryption param"
    )
    parser.add_argument(
        "--addr",
        default="127.0.0.1:37221",
        type=str,
        help="address of client service"
    )

    parser.add_argument(
        "--name",
        default="client_querty",
        type=str,
        help="querty client name"
    )

    args = parser.parse_args()
    print(args)
    controller = PissClientController(args)
    #controller.init_data()
    controller.start_querty()