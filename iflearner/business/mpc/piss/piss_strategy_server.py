from typing import Any, Dict
import numpy as np
from importlib import import_module
from typing import Any, Dict
from loguru import logger

from iflearner.communication.mpc.piss.piss_exception import PissException
from iflearner.communication.mpc.piss import message_type
from iflearner.business.mpc.piss.piss_startegy_base import PissStrategyBase 
from iflearner.communication.mpc.piss import message_type

class PissStrategyServer(PissStrategyBase):
    def __init__(self ,cert_path: str, party_name: str,options) -> None:

        super().__init__(cert_path, party_name, options)
        self._ready_routes: dict = dict()
        self._ready_stubs: dict = dict()
        self._ready_party_name_list = []

    def get_stubs(self):
        return self._stubs

    def get_party_name_list(self):
        return self._party_name_list

    def get_initiator_party_name(self):
        return self._initiator_party_name

    def get_initiator_route(self):
        return self._initiator_route

    def get_initiator_stub(self):
        return self._initiator_stub

    def get_ready_routes(self):
        return self._ready_routes

    def get_ready_party_name_list(self):
        return self._ready_party_name_list

    def update_client_info(self, party_name: str, type: str,route: Any = None):
        if type == message_type.MSG_REGISTER:
            if party_name not in self._party_name_list:
                self._party_name_list.append(party_name)
                self._routes[party_name] = route
                #generate client stub
                self._stubs[party_name] = self.generate_stub(route)
            else:
                raise PissException(PissException.PissResponseCode.AlreadyExistsPartyName, 
                "already exists the same party_name")

        elif type == message_type.MSG_START_QUERY:
            if party_name not in self._party_name_list:
                raise PissException(PissException.PissResponseCode.Unregistered, 
                "client not registered")
            else:
                if party_name not in self._ready_party_name_list:
                    self._ready_party_name_list.append(party_name)
                self._ready_routes[party_name] = self._routes[party_name]
                self._ready_stubs[party_name] = self._stubs[party_name]
                self._initiator_party_name = party_name
                self._initiator_route = self._routes[party_name]
                self._initiator_stub = self._stubs[party_name]

        elif type == message_type.MSG_PARTICIPANTS_READY:
            if party_name not in self._ready_party_name_list:
                self._ready_party_name_list.append(party_name) 
            self._ready_routes[party_name] = self._routes[party_name]
            self._ready_stubs[party_name] = self._stubs[party_name]





