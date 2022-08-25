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
import json
from importlib import import_module
from threading import Thread
from typing import Any, Dict, Union

from flask import Flask
from loguru import logger

from iflearner.business.homo.strategy import (
    fedavg_server,
    fednova_server,
    fedopt_server,
    qfedavg_server,
)
from iflearner.business.homo.strategy.strategy_server import StrategyServer
from iflearner.communication.base import base_server
from iflearner.communication.homo import homo_server, message_type


class AggregateServer:
    """The server processes the requests of all parties according to the usage policy."""

    def __init__(
        self,
        addr: str,
        strategy: Union[str, StrategyServer],
        num_clients: int,
        strategy_params: Dict[str, Any] = {},
        epochs: int = 0,
    ) -> None:
        logger.add("log/server.log", backtrace=True, diagnose=True)
        logger.info(
            f"server address:  {addr}, strategy: {strategy}, client number: {num_clients}, epochs: {epochs}, strategy params: {strategy_params}"
        )

        if isinstance(strategy, str):
            if strategy == message_type.STRATEGY_FEDAVG:
                self._strategy_server = fedavg_server.FedavgServer(
                    num_clients, epochs, False, **strategy_params
                )

            elif strategy == message_type.STRATEGY_SCAFFOLD:
                self._strategy_server = fedavg_server.FedavgServer(
                    num_clients, epochs, True, **strategy_params
                )

            elif strategy == message_type.STRATEGY_FEDOPT:
                if strategy_params.get("opt") is None:
                    raise Exception("expect 'opt' when you use fedopt sever")
                else:
                    opt_type = strategy_params.pop("opt")
                    module = import_module(
                        f"iflearner.business.homo.strategy.opt.{opt_type.lower()}"
                    )
                    opt_class = getattr(module, f"{opt_type}")
                    opt = opt_class(**strategy_params)
                    self._strategy_server = fedopt_server.FedoptServer(
                        num_clients,
                        epochs,
                        opt=opt,
                    )  # type: ignore
                    logger.info(
                        " ".join([f"{k}:{v}" for k, v in strategy_params.items()])
                    )

            elif strategy == message_type.STRATEGY_qFEDAVG:
                self._strategy_server = qfedavg_server.qFedavgServer(
                    num_clients, epochs, **strategy_params
                )  # type: ignore

            elif strategy == message_type.STRATEGY_FEDNOVA:
                self._strategy_server = fednova_server.FedNovaServer(
                    num_clients, epochs, **strategy_params
                )  # type: ignore
        elif isinstance(strategy, StrategyServer):
            self._strategy_server = strategy  # type: ignore

        self._addr = addr

    def run(self) -> None:
        """start server"""

        homo_server_inst = homo_server.HomoServer(self._strategy_server)
        base_server.start_server(self._addr, homo_server_inst)


app = Flask(__name__)
server: AggregateServer = None  # type: ignore


@app.route("/v1/status", methods=["GET"])
def expose_status():
    global server
    return server._strategy_server.clients_to_json()


def http_server(host: str, port: int) -> None:
    app.run(host=host, port=port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num", help="the number of all clients", default=0, type=int
    )
    parser.add_argument("--epochs", help="the total epoch", type=int)
    parser.add_argument(
        "--addr", help="the server address", default="0.0.0.0:50001", type=str
    )
    parser.add_argument(
        "--http_addr", help="the http address", default="0.0.0.0:50002", type=str
    )
    parser.add_argument(
        "--strategy",
        help="the aggregation starategy (FedAvg | Scaffold | FedOpt | qFedAvg | FedNova)",
        default="FedAvg",
        type=str,
    )
    parser.add_argument(
        "--strategy_params",
        help="specify the params of strategy",
        default={},
        type=json.loads,
    )

    args = parser.parse_args()
    global server
    server = AggregateServer(
        args.addr, args.strategy, args.num, args.strategy_params, args.epochs
    )
    Thread(
        target=http_server,
        args=(args.http_addr.split(":")[0], args.http_addr.split(":")[1]),
    ).start()
    server.run()


if __name__ == "__main__":
    main()
