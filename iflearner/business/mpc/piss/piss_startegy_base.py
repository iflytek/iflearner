from abc import ABC
from ast import Return
import code
from enum import IntEnum, auto
import imp
from typing import Any, Dict

import numpy as np

import argparse
import json
from importlib import import_module
from threading import Thread
from typing import Any, Dict, Union
from loguru import logger
import time
import grpc
from iflearner.communication.mpc import piss
import threading


from iflearner.communication.mpc.piss.piss_exception import PissException
from iflearner.communication.base import base_server
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss import piss_pb2, piss_pb2_grpc
from iflearner.communication.base import base_pb2, base_pb2_grpc,base_server,constant


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
