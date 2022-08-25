import argparse
import json

from iflearner.business.homo.aggregate_server import AggregateServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num", help="the number of all clients", default=0, type=int
    )
    parser.add_argument("--epochs", help="the total epoch", type=int)
    parser.add_argument(
        "--addr",
        help="The aggregation server itself listens to the address (used for client connections)",
        default="0.0.0.0:50001",
        type=str,
    )
    parser.add_argument(
        "--http_addr",
        help="Federation training status listening address (for viewing federation training status)",
        default="0.0.0.0:50002",
        type=str,
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

    server = AggregateServer(
        args.addr, args.strategy, args.num, args.strategy_params, args.epochs
    )
    server.run()
