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
from typing import Any, Dict
from visualdl import LogWriter


class Metric:
    """Integrate visualdl to visualize metrics.
    """

    def __init__(self, logdir: str) -> None:
        """Init class with log directory."""

        self._tag_prefix = "train"
        self._logdir = logdir
        self._writers: Dict[str, LogWriter] = dict()
        self._figs: Dict[str, Any] = dict()

    def add(self, name: str, label: str, x: Any, y: Any) -> None:
        """Add a point.

        Args:
            name: The name of metric, eg: acc, loss.
            label: The label of metric, eg: local learning, federated learning.
            x: The x of point, eg: 1, 2, 3...
            y: The y of point, eg: 95.5, 96.0, 96.5...
        """

        if label not in self._writers:
            self._writers[label] = LogWriter(logdir=f"{self._logdir}/{label}", display_name=label)

        self._writers[label].add_scalar(f"{self._tag_prefix}/{name}", y, x)
