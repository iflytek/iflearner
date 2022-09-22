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

from typing import Dict, Tuple, List, Union


def handle_another_step(data: List[Tuple[str, bytes]]) -> None:
    """Handle a step from another role.

    Args:
        data (List[Tuple[str, bytes]]): List data for all role members.
    """
    pass


def handle_own_step() -> Union[Tuple[str, bytes], Dict[str, bytes]]:
    """Handle a own step.

    Returns:
        Union[Tuple[str, bytes], Dict[str, bytes]]:
            Tuple[str, bytes]: If you want to send the same data to all role members, you need to return this (k: role name, v: data).
            Dict[str, bytes]: If you want to send unique data to a specific role member, you need to return this (k: party name, v: data).
    """
    pass


class BaseModel:
    '''Define each step of model training and evaluation.

    You need to split the whole process into steps, then you need to implement it and register it with self._register_own_step.
    In most cases, a step depends on steps completed by other roles. So you need to implement the response of those upstream steps and register it with self._register_another_step.
    '''

    def __init__(self) -> None:
        self._another_steps: Dict[str, handle_another_step] = {}
        self._own_steps: Dict[str, handle_own_step] = {}

    def _register_another_step(self, role: str, step_name: str, func: handle_another_step) -> None:
        """Register a another step handler.

        Args:
            role (str): The target role name.
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

    def handle_upstream(self, role: str, step_name: str, data: List[Tuple[str, bytes]]) -> None:
        """Handle specific upstream step from other role.

        Args:
            role (str): The target role name.
            step_name (str): Unique name for the step.
            data (List[Tuple[str, bytes]]): List data for all role members.
        """
        key = f"{role}.{step_name}"
        assert key in self._another_steps, f"{key} is not implemented."

        self._another_steps[key](data)

    def handle_step(self, step_name: str) -> Union[Tuple[str, bytes], Dict[str, bytes]]:
        """Handle own specific step.

        Args:
            step_name (str): Unique name for the step.

        Returns:
            Union[Tuple[str, bytes], Dict[str, bytes]]:
                Tuple[str, bytes]: If you want to send the same data to all role members, you need to return this (k: role name, v: data).
                Dict[str, bytes]: If you want to send unique data to a specific role member, you need to return this (k: party name, v: data).
        """
        assert step_name in self._own_steps, f"{step_name} is not implemented."

        return self._own_steps[step_name]()
