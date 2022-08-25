import argparse

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

    args = parser.parse_args()

    strategy = message_type.STRATEGY_FEDNOVA

    server = AggregateServer(args.addr, strategy, args.num)
    server.run()
