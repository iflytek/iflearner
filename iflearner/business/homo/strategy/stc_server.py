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
import pickle
import numpy as np
from loguru import logger
from typing import Any, Dict, Optional
from iflearner.communication.homo import homo_pb2, message_type
from iflearner.business.homo.strategy.strategy_server import StrategyServer


class STCServer(StrategyServer):

    def handler_register(self, party_name: str, sample_num: Optional[int] = 0, step_num: int = 0) -> homo_pb2.RegistrationResponse:
        super().handler_register(party_name, sample_num, step_num)

        return homo_pb2.RegistrationResponse(
            strategy=message_type.STRATEGY_STC, parameters=None
        )

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        super().handler_upload_param(party_name, data)

        self._uploaded_num += 1
        if self._uploaded_num == self._num_clients:
            self._uploaded_num = 0
            aggregate_result = dict()
            logger.info(f"Avg params, param num: {len(data.parameters)}")
            for param_name, param_info in data.parameters.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=param_info.shape
                )

                arr_size = math.prod(param_info.shape)
                params = np.zeros(arr_size)

                for v in self._training_clients.values():
                    # params += v["param"][param_name].values
                    sparse_array = self._decode_sparse_array(
                        v["param"][param_name].custom_values, arr_size)
                    params += sparse_array
                    # logger.info(f"Sparse array: {sparse_array}")

                avg_param = params / self._num_clients

                if self._server_param is not None:
                    aggregate_result[param_name].values.extend(
                        [sum(x) for x in zip(
                            avg_param, self._server_param[param_name].values)]
                    )
                else:
                    aggregate_result[param_name].values.extend(avg_param)

            self._server_param = aggregate_result
            self._on_aggregating = True

    def _decode_sparse_array(self, data: bytes, array_size: int) -> np.array:
        """Decode bytes to a sparse array.

        Args:
            data (bytes): The data that dumped by pickle.
            array_size (int): The sparse array size.

        Returns:
            np.array: A sparse array.
        """
        logger.info(
            f"Decode a sparse array, data size: {len(data)}, array size: {array_size}")

        elements = pickle.loads(data)
        mean_value = elements[0]
        positive_horizontal_coordinates = elements[1]
        positive_vertical_coordinates = elements[2]
        negative_horizontal_coordinates = elements[3]
        negative_vertical_coordinates = elements[4]

        uint8_len = np.iinfo(np.uint8).max + 1
        uint16_len = np.iinfo(np.uint16).max + 1
        horizontal_coordinate_len = uint8_len
        if array_size > uint8_len * uint16_len:
            horizontal_coordinate_len = uint16_len

        arr = np.zeros(array_size, dtype=np.float64)
        index = 0
        for item in positive_vertical_coordinates:
            arr[positive_horizontal_coordinates[index] +
                item*horizontal_coordinate_len] = mean_value
            index += 1

        index = 0
        for item in negative_vertical_coordinates:
            arr[negative_horizontal_coordinates[index] +
                item*horizontal_coordinate_len] = -mean_value
            index += 1

        flatten_arr = arr.ravel()
        logger.info(f"Sparse array size: {flatten_arr.size}")

        return flatten_arr
