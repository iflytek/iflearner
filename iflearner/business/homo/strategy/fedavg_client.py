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
from typing import Dict

import numpy as np
import numpy.typing as npt

from iflearner.business.homo.strategy import strategy_client
from iflearner.communication.homo import homo_pb2


class FedavgClient(strategy_client.StrategyClient):
    """Implement the strategy of fedavg on client side."""

    def __init__(self, scaffold: bool = False) -> None:
        super().__init__()
        self._scaffold = scaffold

    def generate_upload_param(
        self, epoch: int, data: Dict[str, npt.NDArray[np.float32]], metrics: Dict[str, float] = None  # type: ignore
    ) -> homo_pb2.UploadParam:
        data = super().generate_upload_param(epoch, data, metrics)

        if self._scaffold:
            plus_dict = dict()
            for k, v in data.parameters.items():  # type: ignore
                plus_dict[k] = homo_pb2.Parameter(shape=v.shape)
                if self._local_c is not None:
                    plus_dict[k].values.extend(
                        [
                            x1
                            - x2
                            + (x3 - x4)
                            / (
                                self._trainer_config["learning_rate"]
                                * self._trainer_config["batch_num"]
                            )
                            for x1, x2, x3, x4 in zip(
                                self._local_c.parameters[k].values,  # type: ignore
                                self._aggregate_c.parameters[k].values,  # type: ignore
                                self._aggregate_result.parameters[k].values,  # type: ignore
                                v.values,
                            )
                        ]
                    )
                else:
                    plus_dict[k].values.extend(
                        [
                            -x4
                            / (
                                self._trainer_config["learning_rate"]
                                * self._trainer_config["batch_num"]
                            )
                            for x4 in v.values
                        ]
                    )

            self._local_c_initial = homo_pb2.UploadParam(parameters=plus_dict)

            updated_data = dict()
            for k, v in self._local_c_initial.parameters.items():  # type: ignore
                updated_data[k + self._gradient_suffix] = homo_pb2.Parameter(
                    shape=v.shape
                )
                if self._local_c is not None:
                    updated_data[k + self._gradient_suffix].values.extend(
                        [
                            x1 - x2
                            for x1, x2 in zip(
                                v.values, self._local_c.parameters[k].values  # type: ignore
                            )
                        ]
                    )
                else:
                    updated_data[k + self._gradient_suffix].values.extend(v.values)

                updated_data[k] = homo_pb2.Parameter(shape=v.shape)
                if self._aggregate_result is not None:
                    updated_data[k].values.extend(
                        [
                            x1 - x2
                            for x1, x2 in zip(
                                data.parameters[k].values,  # type: ignore
                                self._aggregate_result.parameters[k].values,  # type: ignore
                            )
                        ]
                    )
                else:
                    updated_data[k].values.extend(data.parameters[k].values)  # type: ignore

            self._local_c = self._local_c_initial
            self._local_c_initial = None
            return homo_pb2.UploadParam(epoch=data.epoch, parameters=updated_data, metrics=metrics)  # type: ignore

        return data

    def update_param(self, data: homo_pb2.AggregateResult) -> homo_pb2.AggregateResult:
        if not self._scaffold:
            return data

        updated_data = dict()
        for k, v in data.parameters.items():
            updated_data[k] = homo_pb2.Parameter(shape=v.shape)
            if self._local_c is not None:
                updated_data[k].values.extend(
                    [
                        x1 - x2 + x3
                        for (x1, x2, x3) in zip(
                            v.values,
                            self._local_c.parameters[k].values,  # type: ignore
                            self._aggregate_c.parameters[k].values,  # type: ignore
                        )
                    ]
                )
            else:
                updated_data[k].values.extend(v.values)

        return homo_pb2.AggregateResult(parameters=updated_data)
