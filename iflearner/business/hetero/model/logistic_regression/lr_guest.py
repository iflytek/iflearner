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
from typing import Any, List, Dict, Union
from phe import paillier
from iflearner.business.hetero.model.role import Role, host, arbiter
from iflearner.business.hetero.model.base_model import BaseModel


class LRGuest(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self._register_another_step(
            arbiter, "generate_he_keypair", self.received_he_public_key)
        self._register_own_step("empty", self.empty)

    def received_he_public_key(self, data: Dict[str, Any]) -> None:
        for value in data.values():
            public_key = paillier.PaillierPublicKey(value)
            logger.info(f"Public key: {public_key}")
            self._public_key = public_key
            break

    def empty(self) -> Dict[Union[Role, str], Any]:
        pass
