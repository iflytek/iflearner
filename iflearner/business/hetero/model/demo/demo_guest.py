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

from loguru import logger
from typing import List, Tuple
from iflearner.business.hetero.model.role import Role
from iflearner.business.hetero.model.base_model import BaseModel


class DemoGuest(BaseModel):
    """A demo showing the model development.
    """

    def __init__(self) -> None:
        super().__init__()

        self._register_own_step("step1", self.handle_own_step1)
        self._register_own_step("step2", self.handle_own_step2)

        self._register_another_step(
            Role.host, "step1", self.handle_host_step1)
        self._register_another_step(
            Role.arbiter, "step1", self.handle_arbiter_step1)
        self._register_another_step(
            Role.arbiter, "step2", self.handle_arbiter_step2)

    def handle_own_step1(self) -> Tuple[str, bytes]:
        logger.info("Guest step1")
        return Role.host, "Guest step1 completed.".encode("utf-8")

    def handle_own_step2(self):
        logger.info("Guest step2")

    def handle_host_step1(self, data: List[Tuple[str, bytes]]):
        for item in data:
            logger.info(f"{item[0]}, {item[1].decode('utf-8')}")

    def handle_arbiter_step1(self, data: List[Tuple[str, bytes]]):
        for item in data:
            logger.info(f"{item[0]}, {item[1].decode('utf-8')}")

    def handle_arbiter_step2(self, data: List[Tuple[str, bytes]]):
        for item in data:
            logger.info(f"{item[0]}, {item[1].decode('utf-8')}")
