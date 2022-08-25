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
import random
import typing
from datetime import datetime

from numpy import byte

from iflearner.communication.peer import diffie_hellman

p, g = (
    179769313486231590770839156793787453197860296048756011706444423684197180216158519368947833795864925541502180565485980503646440548199239100050792877003355816639229553136239076508735759914822574862575007425302077447712589550957937778424442426617334727629299387668709205606050270810842907692932019128194467627007,
    2,
)
random.seed(datetime.now())
r = random.randint(1, 10000000)
DH_public_key = diffie_hellman.DiffieHellman.encrypt(g, r, p)


class DiffieHellmanInst(object):
    """The Diffie-Hellman instance for generating public key and secret."""

    @staticmethod
    def generate_public_key() -> typing.List[byte]:
        return DH_public_key

    @staticmethod
    def generate_secret(data) -> str:
        return str(diffie_hellman.DiffieHellman.decrypt(data, r, p))
