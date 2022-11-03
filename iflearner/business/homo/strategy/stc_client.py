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

import sys
import pickle
import numpy as np
from loguru import logger
from typing import Dict, Any, List
from iflearner.communication.homo import homo_pb2
from iflearner.business.homo.strategy.strategy_client import StrategyClient


class STCClient(StrategyClient):
    """Implement the STC method base on the paper (https://ieeexplore.ieee.org/document/8889996).
    """

    def __init__(self) -> None:
        super().__init__()

        self._top_fraction: float = 0.1
        self._weights: Dict = None
        self._residuals: Dict = None
        self._enable_residuals: bool = True
        logger.info(
            f"STC client, top fraction: {self._top_fraction}, enable residuals: {self._enable_residuals}")

    def generate_upload_param(self, epoch: int, data: Dict[Any, Any], metrics: Dict[str, float] = None) -> homo_pb2.UploadParam:
        if self._residuals is None:
            self._residuals = {}
            self._weights = {}
            for k, v in data.items():
                self._residuals[k] = np.zeros(v.size)
                self._weights[k] = np.zeros(v.size)

        compressed_data = {}
        for k, v in data.items():
            compressed_data[k] = homo_pb2.Parameter(shape=v.shape)

            ravel_v = v.ravel()

            weight_difference = ravel_v - self._weights[k]
            if self._enable_residuals:
                self._residuals[k] += weight_difference

            self._weights[k] = ravel_v

            out = None
            mean = 0.0
            if self._enable_residuals:
                out, mean = self._compression(self._residuals[k])
                self._residuals[k] -= out
            else:
                out, mean = self._compression(weight_difference)

            sparse_data = self._encode_sparse_array(out, mean)
            compressed_data[k].custom_values = sparse_data

        return homo_pb2.UploadParam(epoch=epoch, parameters=compressed_data, metrics=metrics)

    def aggregate_result(self) -> homo_pb2.AggregateResult:
        for k, v in self._aggregate_result_np.items():
            self._weights[k] = v.flatten()

        return self._aggregate_result_np

    def _compression(self, T: np.array) -> np.array:
        """Compress a array.

        Args:
            T (np.array): The array that needs to be compressed.

        Returns:
            np.array: The compressed array.
        """
        T_abs = np.absolute(T)
        n_top = int(np.ceil(T_abs.size * self._top_fraction))
        topk = T_abs[np.argpartition(T_abs, -n_top)[-n_top:]]
        mean = np.mean(topk)
        min_topk = topk.min()
        out_ = np.where(T >= min_topk, mean, 0.0)
        out = np.where(T <= -min_topk, -mean, out_)
        return out, mean

    def _encode_sparse_array(self, arr: np.array, mean_value: np.float64) -> bytes:
        """Encode a sparse array to bytes.

        Args:
            arr (np.array): A sparse array.

        Returns:
            bytes: The data that dumped by pickle. 
        """
        logger.info(
            f"Encode a sparse array, size: {arr.size}")

        positive_horizontal_coordinates = []
        positive_vertical_coordinates = []
        negative_horizontal_coordinates = []
        negative_vertical_coordinates = []
        horizontal_coordinate_type = np.uint8
        vertical_coordinate_type = np.uint8

        uint8_len = np.iinfo(np.uint8).max + 1
        uint16_len = np.iinfo(np.uint16).max + 1
        if arr.size > uint8_len * uint16_len:
            horizontal_coordinate_type = np.uint16
        if arr.size > uint8_len * uint8_len:
            vertical_coordinate_type = np.uint16

        horizontal_index = 0
        vertical_index = 0
        for item in arr:
            if horizontal_index > np.iinfo(np.uint8).max:
                horizontal_index = 0
                vertical_index += 1
            if item > 0:
                positive_horizontal_coordinates.append(
                    horizontal_coordinate_type(horizontal_index))
                positive_vertical_coordinates.append(
                    vertical_coordinate_type(vertical_index))
            elif item < 0:
                negative_horizontal_coordinates.append(
                    horizontal_coordinate_type(horizontal_index))
                negative_vertical_coordinates.append(
                    vertical_coordinate_type(vertical_index))

            horizontal_index += 1

        my_np_tuple = (mean_value, np.array(positive_horizontal_coordinates),
                       np.array(positive_vertical_coordinates), np.array(negative_horizontal_coordinates), np.array(negative_vertical_coordinates))
        data = pickle.dumps(my_np_tuple)
        logger.info(
            f"After encoding, positive coordinates num: {len(positive_horizontal_coordinates)}, negative coordinates num: {len(negative_horizontal_coordinates)}, the size is {len(data)}")

        return data
