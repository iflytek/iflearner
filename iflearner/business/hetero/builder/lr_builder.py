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

from iflearner.business.hetero.model.role import Role, Guest, Host, Arbiter
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.model.logistic_regression import lr_guest, lr_host, lr_arbiter

from iflearner.business.hetero.builder.model_builder import ModelBuilder


class LRBuilder(ModelBuilder):

    def create_role_model_instance(self, role: Role) -> BaseModel:
        """Create a model instance base on specific role.

        Args:
            role (Role): The role name.

        Returns:
            BaseModel: Return the base class.
        """
        if isinstance(role, Guest):
            return lr_guest.LRGuest()
        elif isinstance(role, Host):
            return lr_host.LRHost()
        elif isinstance(role, Arbiter):
            return lr_arbiter.LRArbiter()

        raise Exception(f"{role} is not existed.")

    def get_role_model_flow_file(self, role: Role) -> str:
        """Get model flow file by role name.

        Args:
            role (Role): The role name.

        Returns:
            str: Return the filename.
        """
        if isinstance(role, Guest):
            return "lr_guest_flow.yaml"
        elif isinstance(role, Host):
            return "lr_host_flow.yaml"
        elif isinstance(role, Arbiter):
            return "lr_arbiter_flow.yaml"

        raise Exception(f"{role} is not existed.")
