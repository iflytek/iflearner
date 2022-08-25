import argparse
import json

from iflearner.business.homo.aggregate_server import AggregateServer
from iflearner.communication.homo import message_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num", help="the number of all clients", type=int, default=1
    )

    parser.add_argument(
        "--addr", help="the server address", default="0.0.0.0:50001", type=str
    )
    parser.add_argument(
        "--strategy_params",
        help="specify the params of strategy",
        type=json.loads,
        default='{"learning_rate":1, "betas":[0.9,0.99], "t":0.1, "opt":"FedAdam"}',
    )

    args = parser.parse_args()

    strategy = message_type.STRATEGY_FEDOPT

    server = AggregateServer(args.addr, strategy, args.num, args.strategy_params)
    server.run()
