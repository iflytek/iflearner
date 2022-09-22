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

from typing import List, Tuple
from iflearner.business.hetero.model.role import Role
from iflearner.business.hetero.model.base_model import BaseModel


class LRArbiter(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self._register_own_step("step1", self.handle_own_step1)
        self._register_own_step("step2", self.handle_own_step2)

        self._register_another_step(
            Role.host, "step2", self.handle_host_step2)

    def handle_own_step1(self) -> Tuple[str, bytes]:
        print("Arbiter step1")
        return Role.guest, "Arbiter step1 completed.".encode("utf-8")

    def handle_own_step2(self):
        print("Arbiter step2")
        return Role.guest, "Arbiter step2 completed.".encode("utf-8")

    def handle_host_step2(self, data: List[Tuple[str, bytes]]):
        for item in data:
            print(item[0], item[1].decode("utf-8"))
