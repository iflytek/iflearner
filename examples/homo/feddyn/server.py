import argparse
from statistics import mode

from feddyn_server import FedDynServer
from torchvision import models

from iflearner.business.homo.aggregate_server import AggregateServer
from iflearner.business.homo.strategy.opt import FedAdam
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

    model = models.__dict__["resnet18"](pretrained=False)
    model.train()
    params = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.numpy()

    strategy = FedDynServer(args.num, learning_rate=1, alpha=1, params=params)

    server = AggregateServer(args.addr, strategy, args.num)
    server.run()
