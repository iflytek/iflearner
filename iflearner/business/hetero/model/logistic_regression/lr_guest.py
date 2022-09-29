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

import numpy as np
from phe import paillier
from loguru import logger
from typing import Any, List, Dict, Union
from iflearner.business.hetero.model.role import Role, host, arbiter
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.model.logistic_regression.dataset import get_guest_data


class LRGuest(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self._x, self._y = get_guest_data()
        self._weights = np.zeros(self._x.shape[1])

        self._register_another_step(
            arbiter, "generate_he_keypair", self.received_he_public_key)
        self._register_another_step(
            host, "calc_host_partial_result", self.received_host_partial_result)
        self._register_another_step(
            arbiter, "decrypt_guest_data", self.received_weights)

        self._register_own_step("calc_guest_partial_result",
                                self.calc_guest_partial_result)
        self._register_own_step(
            "calc_final_result_with_host", self.calc_final_result_with_host)

    def set_hyper_params(self, hyper_params: Any) -> None:
        """Set hyper params.

        Args:
            hyper_params (Any): Details of the hyper params.
        """
        self._lambda = 10
        self._lr = 0.05
        if hyper_params is not None:
            self._lr = hyper_params.get("lr", 0.05)
            self._lambda = hyper_params.get("lambda", 10)

        logger.info(f"lr: {self._lr}, lambda: {self._lambda}")

    def received_he_public_key(self, data: Dict[str, Any]) -> None:
        """Save the HE public key received from the arbiter.

        Args:
            data (Dict[str, Any]): Arbiter party name and public key.
        """
        public_key = paillier.PaillierPublicKey(list(data.values())[0])
        logger.info(f"Public key: {public_key}")
        self._public_key = public_key

    def calc_guest_partial_result(self) -> Dict[Union[Role, str], Any]:
        """Calculate your own partial results.

        Returns:
            Dict[Union[Role, str], Any]: Return HE-encrypted data to the host.
        """
        self._z_b = np.dot(self._x, self._weights)
        u_b = 0.25 * self._z_b - self._y + 0.5
        self._encrypted_u_b = np.asarray(
            [self._public_key.encrypt(x) for x in u_b])
        return {host: self._encrypted_u_b}

    def received_host_partial_result(self, data: Dict[str, Any]) -> None:
        """Save the host partial result.

        Args:
            data (Dict[str, Any]): Host party name and its data.
        """
        self._encrypted_u_a, self._encrypted_z_a_square = list(data.values())[
            0]

    def calc_final_result_with_host(self) -> Dict[Union[Role, str], Any]:
        """Calculate the final result combined with the host.

        Returns:
            Dict[Union[Role, str], Any]: Return the encrypted result to the arbiter.
        """
        encrypted_u = self._encrypted_u_a + self._encrypted_u_b
        encrypted_dJ_b = self._x.T.dot(
            encrypted_u) + self._lambda * self._weights
        self._mask = np.random.rand(len(encrypted_dJ_b))
        encrypted_masked_dJ_b = encrypted_dJ_b + self._mask

        encrypted_z = 4*self._encrypted_u_a + self._z_b
        encrypted_loss = np.sum((0.5-self._y)*encrypted_z + 0.125 * self._encrypted_z_a_square +
                                0.125*self._z_b * (encrypted_z+4*self._encrypted_u_a))

        return {arbiter: (encrypted_masked_dJ_b, encrypted_loss, self._x.shape[0])}

    def received_weights(self, data: Dict[str, Any]) -> None:
        """Received weights from the arbiter.

        Args:
            data (Dict[str, Any]): The decrypted data from the arbiter.
        """
        masked_dJ_b = list(data.values())[0]
        dJ_b = masked_dJ_b - self._mask
        self._weights = self._weights - self._lr * dJ_b / len(self._x)
