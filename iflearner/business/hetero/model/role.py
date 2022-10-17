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

from abc import ABC, abstractmethod


class Role(ABC):

    @abstractmethod
    def __str__(self) -> str:
        pass


class Guest(Role):
    def __str__(self) -> str:
        return "guest"


class Host(Role):
    def __str__(self) -> str:
        return "host"


class Arbiter(Role):
    def __str__(self) -> str:
        return "arbiter"


guest = Guest()
host = Host()
arbiter = Arbiter()


def role_class(name: str) -> Role:
    if name == str(guest):
        return guest
    elif name == str(host):
        return host
    elif name == str(arbiter):
        return arbiter
    else:
        raise Exception(f"Role {name} is not existed.")
