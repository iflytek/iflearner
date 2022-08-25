## To build your own homo strategy.

### 1.Build Your Server

In order to build Federate Learning Server, you should inherit `iflearner.business.homo.strategy.strategy_server.StrategyServer` and implement all the abstract methods.

- `handler_register`: Handle the message of MSG_REGISTER from the client.
- `handler_client_ready`: Handle the message of MSG_CLIENT_READY from the client.
- `handler_upload_param`: Handle the message of MSG_UPLOAD_PARAM from the client. The content of this message is determined by the client's Trainner.get() method. 
- `get_client_notification`: Get the notification information of the specified client.


The following is a simple example. For more details, you can refer to the fedavg implementation, or contact us.

```python
class StrategyServer():
    """Implement the strategy of server."""

    def __init__(self) -> None:
        self._custom_handlers: Dict[str, Any] = dict()

    @property
    def custom_handlers(self) -> Dict[str, Any]:
        return self._custom_handlers

    def handler_register(
        self,
        party_name: str,
        sample_num: Optional[int] = None,
        step_num: Optional[int] = None,
    ) -> None:
        """Handle the message of MSG_REGISTER from the client."""
        pass

    def handler_client_ready(self, party_name: str) -> None:
        """Handle the message of MSG_CLIENT_READY from the client."""
        pass

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        """Handle the message of MSG_UPLOAD_PARAM from the client."""
        pass

    def get_client_notification(self, party_name: str) -> Tuple[str, Any]:
        """Get the notification information of the specified client."""
        pass

class MyStrategyServer(StrategyServer):
    ''''''

strategy = MyStrategyServer(...)
server = AggregateServer(args.addr, strategy, args.num)
server.run()

```

### 2. Build Your Client

In order to build Federate Learning Client, you should inherit <code>iflearner.business.homo.strategy.strategy_client.StrategyClient</code> and implement all the abstract methods.


```python
class StrategyClient(ABC):
    """Implement the strategy of client."""

    class Stage(IntEnum):
        """Enum the stage of client."""

        Waiting = auto()
        Training = auto()
        Setting = auto()

    def __init__(self) -> None:
        self._custom_handlers: Dict[str, Any] = dict()
        self._trainer_config: Dict[str, Any] = dict()

    @property
    def custom_handlers(self) -> Dict[str, Any]:
        return self._custom_handlers

    def set_trainer_config(self, config: Dict[str, Any]) -> None:
        self._trainer_config = config

    @abstractmethod
    def generate_registration_info(self) -> None:
        """Generate the message of MSG_REGISTER."""
        pass

    @abstractmethod
    def generate_upload_param(self, epoch: int, data: Dict[Any, Any]) -> Any:
        """Generate the message of MSG_UPLOAD_PARAM."""
        pass

    @abstractmethod
    def update_param(self, data: homo_pb2.AggregateResult) -> homo_pb2.AggregateResult:
        """Update the parameter during training."""
        pass

    @abstractmethod
    def handler_aggregate_result(self, data: homo_pb2.AggregateResult) -> None:
        """Handle the message of MSG_AGGREGATE_RESULT from the server."""
        pass

    @abstractmethod
    def handler_notify_training(self) -> None:
        """Handle the message of MSG_NOTIFY_TRAINING from the server."""
        pass

class MyStrategyClient(StrategyClient):
    ''''''

mnist = Mnist()
controller = Controller(args, mnist, MyStrategyClient())
controller.run()

```

## Example

In this example, we will implement [FedDyn](https://arxiv.org/abs/2111.04263) step by step. Please refer to the original paper for FedDyn's algorithm flow. The complete source code 
reference for this example [FedDyn](https://git.iflytek.com/TURING/iflearner/-/tree/master/examples/homo/feddyn).
### Build FedDyn Server

Inherit `iflearner.business.homo.strategy.strategy_server.StrategyServer` class and override and implement some methods according to the algorithm flow. In FedDyn, we just need to override `handler_upload_param` and `__init__`.

```python
from time import sleep
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import dtype, nn
from torch.nn import functional as F

from iflearner.business.homo.strategy import strategy_server
from iflearner.communication.homo import homo_pb2, message_type
from iflearner.communication.homo.homo_exception import HomoException


class FedDynServer(strategy_server.StrategyServer):
    """Implement the strategy of feddyn on server side."""

    """
    num_clients: client numuber
    alpha: a static coefficient 
    """

    def __init__(
        self,
        num_clients: int,
        learning_rate=0.1,
        alpha=0.1,
        params: Dict[str, np.ndarray] = None,
    ) -> None:
        super().__init__()

        self._num_clients = num_clients
        self._lr = learning_rate
        self._alpha = alpha
        self._params = params

        logger.info(f"num_clients: {self._num_clients}")

        self._training_clients: dict = {}
        self._server_param = None
        self._ready_num = 0
        self._uploaded_num = 0
        self._aggregated_num = 0
        self._on_aggregating = False
        self._clients_samples: dict = {}

        self._h = {
            name: np.zeros_like(p).reshape(-1) for name, p in self._params.items()
        }

    def handler_upload_param(self, party_name: str, data: homo_pb2.UploadParam) -> None:
        logger.info(f"Client: {party_name}, epoch: {data.epoch}")

        if party_name not in self._training_clients:
            raise HomoException(
                HomoException.HomoResponseCode.Forbidden, "Client not notified."
            )

        self._training_clients[party_name]["param"] = data.parameters
        self._uploaded_num += 1
        if self._uploaded_num == self._num_clients:
            self._uploaded_num = 0
            aggregate_result = dict()
            grad = dict()

            logger.info(f"Faddyn params, param num: {len(data.parameters)}")

            for param_name, param_info in data.parameters.items():
                aggregate_result[param_name] = homo_pb2.Parameter(
                    shape=param_info.shape
                )
                params = []
                for v in self._training_clients.values():
                    params.append(v["param"][param_name].values)

                avg_param = [sum(x) * (1 / self._num_clients) for x in zip(*params)]
                grad[param_name] = np.array(avg_param, dtype="float32") - self._params[
                    param_name
                ].reshape((-1))
                self._h[param_name] = (
                    self._h[param_name] - self._alpha * grad[param_name]
                )
                self._params[param_name] = (
                    np.array(avg_param, dtype="float32")
                    - (1 / self._alpha) * self._h[param_name]
                ).reshape(param_info.shape)

                aggregate_result[param_name].values.extend(
                    self._params[param_name].reshape(-1).tolist()
                )

            self._server_param = aggregate_result  # type: ignore
            self._on_aggregating = True

```

###  Build Client

According to the FedDyn algorithm flow, we need to override the `Trainner.fit` method

```python
def fit(self, epoch):
        self._old_weights = deepcopy(self._model.state_dict())
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(self._train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="{} Epoch: [{}]".format(self._args.name, epoch),
        )

        # switch to train mode
        self._model.train()

        end = time.time()
        for _ in range(1):
            for i, (images, target) in enumerate(self._train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                if self._args.gpu is not None:
                    images = images.cuda(self._args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(self._args.gpu, non_blocking=True)

                # compute output
                output = self._model(images)
                loss = self._criterion(output, target)
                linear_penalty = sum(
                    [
                        torch.sum((p * self._old_grad[name])).cpu().detach().numpy()
                        for name, p in self._model.named_parameters()
                        if p.requires_grad
                    ]
                )

                quad_penalty = sum(
                    [
                        F.mse_loss(p, self._old_weights[name], reduction="sum")
                        .cpu()
                        .detach()
                        .numpy()
                        for name, p in self._model.named_parameters()
                        if p.requires_grad
                    ]
                )

                loss += quad_penalty * self._alpha / 2
                loss -= linear_penalty

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # compute gradient and do SGD step
                self._optimizer.zero_grad()
                loss.backward()

                if self._scaffold:
                    g = yield self.get(self.ParameterType.ParameterGradient)
                    self.set(
                        homo_pb2.AggregateResult(parameters=g.parameters),
                        self.ParameterType.ParameterGradient,
                    )

                self._optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self._args.print_freq == 0:
                    progress.display(i)
                self._old_grad = {
                    name: p.grad for name, p in self._model.named_parameters()
                }
```