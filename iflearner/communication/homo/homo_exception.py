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
from enum import IntEnum, auto

from iflearner.communication.base import base_exception


class HomoException(base_exception.BaseException):
    class HomoResponseCode(IntEnum):
        """Define response code"""

        BadRequest = auto()
        Unauthorized = auto()
        Forbidden = auto()
        Conflict = auto()
        InternalError = auto()

    def __init__(self, code: HomoResponseCode, message: str) -> None:
        super().__init__(
            code.value, f"{code.__class__.__name__}.{code.name} - {message}"
        )
