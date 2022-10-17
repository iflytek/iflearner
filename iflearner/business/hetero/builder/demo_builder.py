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

from iflearner.business.hetero.model.role import Role
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.model.demo import demo_guest, demo_host, demo_arbiter

from iflearner.business.hetero.builder.model_builder import ModelBuilder


class DemoBuilder(ModelBuilder):

    def create_role_model_instance(self, role: str) -> BaseModel:
        """Create a model instance base on specific role.

        Args:
            role (str): The role name.

        Returns:
            BaseModel: Return the base class.
        """
        if role == Role.guest:
            return demo_guest.DemoGuest()
        elif role == Role.host:
            return demo_host.DemoHost()
        elif role == Role.arbiter:
            return demo_arbiter.DemoArbiter()

        raise Exception(f"{role} is not existed.")

    def get_role_model_flow_file(self, role: str) -> str:
        """Get model flow file by role name.

        Args:
            role (str): The role name.

        Returns:
            str: Return the filename.
        """
        if role == Role.guest:
            return "demo_guest_flow.yaml"
        elif role == Role.host:
            return "demo_host_flow.yaml"
        elif role == Role.arbiter:
            return "demo_arbiter_flow.yaml"

        raise Exception(f"{role} is not existed.")
