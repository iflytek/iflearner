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
from .fedavg_server import FedavgServer
from .fedopt_server import FedoptServer
from .opt import *
from .qfedavg_server import qFedavgServer

__all__ = [
    "FedoptServer",
    "FedavgServer",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
    "qFedavgServer",
]
