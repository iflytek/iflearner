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

import yaml
import argparse
from typing import Any, Tuple, Dict, List


class Parser:
    """Parse various configuration files.

    Attributes:
        party_name (str): The party name.
        model_name (str): The model name.
        role_name (str): The role name.
        model_flow (dict): The model flow information.
        network_config (Tuple[str, str, Dict[str, List[Tuple[str, str]]]]): The necessary network configuration.
    """

    party_name: str = None
    model_name: str = None
    role_name: str = None
    model_flow: Dict[str, Any] = dict()
    network_config: Tuple[str, str, str,
                          Dict[str, List[Tuple[str, str]]]] = tuple()

    def __init__(self) -> None:
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument(
            "--name", type=str, required=True, help="The name of current instance.")
        argument_parser.add_argument(
            "--model", type=str, required=True, help="The name of model.")
        args = argument_parser.parse_args()
        self.party_name = args.name
        self.model_name = args.model

    def parse_model_flow_file(self, path: str):
        """Load model flow file and parse the flow.

        Args:
            path (str): The path of file.
        """
        with open(path, "r") as stream:
            self.model_flow = yaml.safe_load(stream)

    def parse_task_configuration_file(self, path: str = "task.yaml"):
        """Load task configuration file and parse the configuration.

        Args:
            path (str): The path of file.

        Raise:
            Exception(f"{self.party_name} is not existed."): self.party_name is not existed.
        """
        with open(path, "r") as stream:
            object = yaml.safe_load(stream)
            parties: Dict[str, List[Tuple[str, str]]] = dict()
            network_config = []
            for role_name, insts in object.items():
                role_parties: List[Tuple[str, str]] = []
                is_same_role = False
                for inst in insts:
                    if inst["name"] == self.party_name:
                        network_config.append(inst["addr"])
                        network_config.append(inst["name"])
                        self.role_name = role_name
                        is_same_role = True
                        break
                    else:
                        role_parties.append((inst["name"], inst["addr"]))

                if not is_same_role:
                    parties[role_name] = role_parties

            network_config.append(parties)
            self.network_config = tuple(network_config)
            if self.role_name is None:
                raise Exception(f"{self.party_name} is not existed.")
