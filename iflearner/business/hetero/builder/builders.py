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
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.builder.demo_builder import DemoBuilder
from iflearner.business.hetero.builder.lr_builder import LRBuilder
from iflearner.business.hetero.builder.Xgboost_builder import XGboostBuilder

Builders: Dict[str, BaseModel] = {
    "demo": DemoBuilder(),
    "logistic_regression": LRBuilder(),
    "Xgboost": XGboostBuilder()
}
