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
from iflearner.business.hetero.model.role import Role
from iflearner.business.hetero.model.base_model import BaseModel


class ModelBuilder(ABC):
    """Build a model instance base on the role you specify.
    """

    @abstractmethod
    def create_role_model_instance(self, role: Role) -> BaseModel:
        """Create a model instance base on specific role.

        Args:
            role (Role): The role name.

        Returns:
            BaseModel: Return the base class.
        """
        pass

    @abstractmethod
    def get_role_model_flow_file(self, role: Role) -> str:
        """Get model flow file by role name.

        Args:
            role (Role): The role name.

        Returns:
            str: Return the filename.
        """
        pass
