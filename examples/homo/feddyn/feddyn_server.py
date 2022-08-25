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
