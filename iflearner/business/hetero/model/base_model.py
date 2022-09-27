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

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Any
from iflearner.business.hetero.model.role import Role


def handle_another_step(data: Dict[str, Any]) -> None:
    """Handle a step from another role.

    Args:
        data (Dict[str, Any]): List data for all role members. (str: party name, Any: data)
    """
    pass


def handle_own_step() -> Dict[Union[Role, str], Any]:
    """Handle a own step.

    Returns:
        Dict[Union[Role, str], Any]: Return each target and its data.
            Union[Role, str]: The role class or party name.
            Any: Return a python object, which we will serialize to bytes using pickle.dumps.
    """
    pass


class BaseModel(ABC):
    '''Define each step of model training and evaluation.

    You need to split the whole process into steps, then you need to implement it and register it with self._register_own_step.
    In most cases, a step depends on steps completed by other roles. So you need to implement the response of those upstream steps and register it with self._register_another_step.
    '''

    def __init__(self) -> None:
        self._another_steps: Dict[str, handle_another_step] = {}
        self._own_steps: Dict[str, handle_own_step] = {}

    @abstractmethod
    def set_hyper_params(self, hyper_params: Any) -> None:
        """Set hyper params.

        Args:
            hyper_params (Any): Details of the hyper params.
        """
        pass

    def _register_another_step(self, role: Role, step_name: str, func: handle_another_step) -> None:
        """Register a another step handler.

        Args:
            role (Role): The target role name.
            step_name (str): Unique name for the step.
            func (handle_another_step): The handler you implement.
        """
        self._another_steps[f"{role}.{step_name}"] = func

    def _register_own_step(self, step_name: str, func: handle_own_step) -> None:
        """Register a own step handler.

        Args:
            step_name (str): Unique name for the step.
            func (handle_own_step): The handler you implement.
        """
        self._own_steps[step_name] = func

    def handle_upstream(self, role: Role, step_name: str, data: Dict[str, Any]) -> None:
        """Handle specific upstream step from other role.

        Args:
            role (Role): The target role name.
            step_name (str): Unique name for the step.
            data (Dict[str, Any]): List data for all role members.
        """
        key = f"{role}.{step_name}"
        assert key in self._another_steps, f"{key} is not implemented."

        self._another_steps[key](data)

    def handle_step(self, step_name: str) -> Dict[Union[Role, str], Any]:
        """Handle own specific step.

        Args:
            step_name (str): Unique name for the step.

        Returns:
            Dict[Union[Role, str], Any]: Return each target and its data.
                Union[Role, str]: The role class or party name.
                Any: Return a python object, which we will serialize to bytes using pickle.dumps.
        """
        assert step_name in self._own_steps, f"{step_name} is not implemented."

        return self._own_steps[step_name]()
