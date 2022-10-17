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

import json
import yaml
import argparse
from typing import Any, Tuple, Dict, List
import sys
sys.path.append("/data1/jhjiang/iflearner/")
from iflearner.business.hetero.model.role import Role, role_class


class Upstream:
    """Python class for representing the Upstream field.
    """

    def __init__(self, role: str, step: str) -> None:
        self.role = role
        self.step = step

    def __str__(self) -> str:
        return f"Role: {self.role}, Step: {self.step}"


class InitStep:
    """Python class for representing the InitStep field.
    """

    def __init__(self, name: str, upstreams: List[Upstream] = None, virtual: bool = False) -> None:
        self.name = name
        self.upstreams = upstreams
        self.virtual = virtual

    def __str__(self) -> str:
        format_string = f"Name: {self.name}, Virtual: {self.virtual}, Upstreams: ["

        if self.upstreams is not None:
            for upstream in self.upstreams:
                format_string += f"<{upstream}>"

        return format_string + "]"


Step = InitStep


class ModelFlow:
    """Python class for representing the model flow file.
    """

    def __init__(self, role: str, init_steps: List[InitStep] = None, steps: List[Step] = None) -> None:
        self.role = role
        self.init_steps = init_steps
        self.steps = steps

    def __str__(self) -> str:
        format_string = f"\nRole: {self.role} \n"
        format_string += f"InitSteps: \n"
        if self.init_steps is not None:
            for init_step in self.init_steps:
                format_string += f"  {init_step} \n"

        format_string += f"Steps: \n"
        if self.steps is not None:
            for step in self.steps:
                format_string += f"  {step} \n"
        return format_string


class Parser:
    """Parse various configuration files.

    Attributes:
        party_name (str): The party name.
        model_name (str): The model name.
        role_name (str): The role name.
        epochs (int): Total epochs.
        hyper_params (json object): The hyper parameters.
        model_flow (ModelFlow): The model flow information.
        network_config (Tuple[str, str, Dict[str, List[Tuple[str, str]]]]): The necessary network configuration.
    """

    party_name: str = None
    model_name: str = None
    role_name: Role = None
    model_flow: ModelFlow = None
    network_config: Tuple[str, str, str,
                          Dict[str, List[Tuple[str, str]]]] = tuple()

    def __init__(self) -> None:
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument(
            "--name", type=str, required=True, help="The name of current instance.")
        argument_parser.add_argument(
            "--model", type=str, required=True, help="The name of model.")
        argument_parser.add_argument(
            "--epochs", type=int, default=1, help="Total epochs.")
        argument_parser.add_argument(
            "--hyper_params", type=json.loads, help="The model hyper params.")

        args = argument_parser.parse_args()
        self.party_name = args.name
        self.model_name = args.model
        self.epochs = args.epochs
        self.hyper_params = args.hyper_params

    def parse_model_flow_file(self, path: str):
        """Load model flow file and parse the flow.

        Args:
            path (str): The path of file.
        """
        with open(path, "r") as stream:
            data = yaml.safe_load(stream)
            init_steps = []
            if "init_steps" in data and data["init_steps"] is not None:
                for init_step in data["init_steps"]:
                    virtual = False
                    if "virtual" in init_step:
                        virtual = init_step["virtual"]

                    upstreams = []
                    if "upstreams" in init_step and init_step["upstreams"] is not None:
                        for upstream in init_step["upstreams"]:
                            upstreams.append(
                                Upstream(role=upstream["role"], step=upstream["step"]))
                    init_steps.append(
                        InitStep(name=init_step["name"], virtual=virtual, upstreams=upstreams))

            if "steps" in data and data["steps"] is not None:
                steps = []
                for step in data["steps"]:
                    virtual = False
                    if "virtual" in step:
                        virtual = step["virtual"]

                    upstreams = []
                    if "upstreams" in step and step["upstreams"] is not None:
                        for upstream in step["upstreams"]:
                            upstreams.append(
                                Upstream(role=upstream["role"], step=upstream["step"]))
                    steps.append(
                        Step(name=step["name"], virtual=virtual, upstreams=upstreams))

            self.model_flow = ModelFlow(
                role=data["role"], init_steps=init_steps, steps=steps)

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
                        self.role_name = role_class(role_name)
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
