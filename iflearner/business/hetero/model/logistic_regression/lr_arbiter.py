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

import math
import numpy as np
from loguru import logger
from phe import paillier
from typing import Any, List, Dict, Union

from iflearner.business.hetero.model.role import Role, guest, host
from iflearner.business.hetero.model.base_model import BaseModel


class LRArbiter(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        # self._register_another_step(guest, "calc_final_result_with_host", self.received_guest_encrypted_data)
        # self._register_another_step(host, "calc_final_result_with_guest", self.received_host_encrypted_data)

        # self._register_own_step("generate_he_keypair", self.generate_he_keypair)
        # self._register_own_step("decrypt_guest_data", self.decrypt_guest_data)
        # self._register_own_step("decrypt_host_data", self.decrypt_host_data)
        
    def set_hyper_params(self, hyper_params: Any) -> None:
        """Set hyper params.

        Args:
            hyper_params (Any): Details of the hyper params.
        """
        super().set_hyper_params(hyper_params)

    def generate_he_keypair(self) -> Dict[Union[Role, str], Any]:
        """Generate HE public key and private key.

        Bind:
            step: generate_he_keypair

        Returns:
            Dict[Union[Role, str], Any]: Return the HE public key to the guest and host.
        """
        public_key, private_key = paillier.generate_paillier_keypair()
        logger.info(f"Public key: {public_key}")
        self._private_key = private_key
        return {guest: public_key.n, host: public_key.n}
    
    def received_guest_encrypted_data(self, data: Dict[str, Any]) -> None:
        """Save encrypted data from the guest.

        Bind:
            step: calc_final_result_with_host
            role: guest

        Args:
            data (Dict[str, Any]): Guest party name and encrypted data.
        """
        self._encrypted_masked_dJ_b, encrypted_loss, shape = list(data.values())[0]
        loss = self._private_key.decrypt(encrypted_loss) / shape + math.log(2)
        logger.info(f"Loss: {loss}")
    
    def received_host_encrypted_data(self, data: Dict[str, Any]) -> None:
        """Save encrypted data from the host.

        Bind:
            step: calc_final_result_with_guest
            role: host

        Args:
            data (Dict[str, Any]): Host party name and encrypted data.
        """
        self._encrypted_masked_dJ_a = list(data.values())[0]
    
    def decrypt_guest_data(self) -> Dict[Union[Role, str], Any]:
        """Decrypt guest data.

        Bind:
            step: decrypt_guest_data

        Returns:
            Dict[Union[Role, str], Any]: Return guest role name and its decrypted data.
        """
        masked_dJ_b = np.asarray([self._private_key.decrypt(x) for x in self._encrypted_masked_dJ_b])
        return {guest: masked_dJ_b}
    
    def decrypt_host_data(self) -> Dict[Union[Role, str], Any]:
        """Decrypt host data.

        Bind:
            step: decrypt_host_data

        Returns:
            Dict[Union[Role, str], Any]: Return host role name and its decrypted data.
        """
        masked_dJ_a = np.asarray([self._private_key.decrypt(x) for x in self._encrypted_masked_dJ_a])
        return {host: masked_dJ_a}
    
    
    