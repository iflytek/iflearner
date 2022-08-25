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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    default="client",
    type=str,
    help="name of client",
)

parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="number of total epochs to run",
)

parser.add_argument(
    "--server",
    default="localhost:50001",
    type=str,
    help="address of aggerating server",
)

parser.add_argument(
    "--cert",
    default=None,
    type=str,
    help="path of server SSL cert"
    """use secure channel to connect to server if not none""",
)

parser.add_argument(
    "--enable-ll",
    default=0,
    type=int,
    help="enable local training (1 | 0)",
)

parser.add_argument(
    "--peers",
    default=None,
    type=str,
    help="enabled SMPC if the argument had specified "
    """all clients' addresses and use semicolon separate all addresses """
    "First one is your own address. ",
)

parser.add_argument(
    "--peer-cert",
    default=None,
    type=str,
    help="path of party SSL cert"
    """use secure channel to connect to other parties if not none""",
)
