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

import time
from threading import Thread
from loguru import logger
from typing import Dict, List, Tuple, Union
from iflearner.business.hetero.model.role import Role
from iflearner.communication.base.base_server import start_server
from iflearner.communication.hetero.hetero_client import HeteroClient
from iflearner.communication.hetero.hetero_server import HeteroServer


class HeteroNetwork:

    def __init__(self, addr: str, own_party_name: str, parties: Dict[str, List[Tuple[str, str]]]) -> None:
        """Initialize the network with some necessary information.

        Args:
            addr (str): The own listening address.
            own_party_name (str): Own party name.
            parties (Dict[str, List[Tuple[str, str]]]): All parties information.
        """
        self._parties_index_role_name = dict()
        self._parties_index_party_name = dict()

        party_with_role = dict()
        for role, members in parties.items():
            clients = []
            for member in members:
                cli = HeteroClient(member[1], own_party_name)
                clients.append(cli)
                self._parties_index_party_name[member[0]] = cli
                party_with_role[member[0]] = role

            self._parties_index_role_name[role] = clients

        self._server = HeteroServer(party_with_role)
        self._thread = Thread(target=start_server, args=(addr, self._server))
        self._thread.start()

    def pull(self, role: str, step_name: str) -> Dict[str, bytes]:
        """Pull messages for a specific role step.
        A role may contain several members and we will wait until all members have uploaded their data.
        Returns none if someone is not ready yet.

        Args:
            role (str): The target role name.
            step_name (str): The specific step name.

        Raises:
            Exception(f"{role} is not existed."): Role is not existed.

        Returns:
            Dict[str, bytes]: Data received from all role members.
        """
        if role not in self._parties_index_role_name:
            raise Exception(f"{role} is not existed.")

        key = f"{role}.{step_name}"
        if key not in self._server.messages:
            return None

        if len(self._server.messages[key]) != len(self._parties_index_role_name[role]):
            return None

        return self._server.messages.pop(key)

    def push(self, name: Union[Role, str], step_name: str, data: bytes) -> None:
        """Push a message to a specific destination, which you can specify using a role or party name.
        If you use a role name, we will send the data to all role members.
        If you use a party name, we will only send the data to the specific target.

        Args:
            name (Union[Role, str]): The role or party name.
            step_name (str): Current step name.
            data (bytes): The data needs to be sent.

        Raises:
            Exception(f"Role {name} is not existed."): Role is not existed.
            Exception(f"Party {name} is not existed."): Party is not existed.
        """
        logger.info(
            f"Post message, name: {name}, step: {step_name}, data length: {len(data)}")
        if isinstance(name, Role):
            name = str(name)
            if name not in self._parties_index_role_name:
                raise Exception(f"Role {name} is not existed.")

            for client in self._parties_index_role_name[name]:
                while True:
                    try:
                        client.post(step_name, data)
                        break
                    except Exception as e:
                        logger.warning(e)
                        time.sleep(3)
        else:
            if name not in self._parties_index_party_name:
                raise Exception(f"Party {name} is not existed.")

            self._parties_index_party_name[name].post(step_name, data)
