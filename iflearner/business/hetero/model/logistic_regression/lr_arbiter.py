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
from phe import paillier
from typing import Any, List, Dict, Union

from iflearner.business.hetero.model.role import Role, guest, host
from iflearner.business.hetero.model.base_model import BaseModel


class LRArbiter(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self._register_own_step("generate_he_keypair",
                                self.generate_he_keypair)

    def generate_he_keypair(self) -> Dict[Union[Role, str], Any]:
        public_key, private_key = paillier.generate_paillier_keypair()
        logger.info(f"Public key: {public_key}")
        self._private_key = private_key
        return {guest: public_key.n, host: public_key.n}
    