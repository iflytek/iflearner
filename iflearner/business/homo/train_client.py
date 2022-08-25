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
import os
import time
import types
from threading import Thread

from loguru import logger

from iflearner.business.homo.strategy import (
    fedavg_client,
    fednova_client,
    fedopt_client,
    qfedavg_client,
)
from iflearner.business.homo.strategy.strategy_client import StrategyClient
from iflearner.business.homo.trainer import Trainer
from iflearner.business.util.metric import Metric
from iflearner.communication.base import base_server
from iflearner.communication.homo import homo_client, homo_pb2, message_type
from iflearner.communication.peer import diffie_hellman_inst, peer_client, peer_server


class Controller:
    """Control the training logic of the client."""

    def __init__(self, args: argparse.Namespace, trainer: Trainer) -> None:
        logger.add(f"log/{args.name}.log", backtrace=True, diagnose=True)
        self._args = args
        self._trainer = trainer
        self._network_client = homo_client.HomoClient(
            self._args.server, self._args.name, self._args.cert
        )
        self._party_name = self._args.name
        self._sum_random_value = 0.0
        self._epoch = 1
        self._local_training = "LT"
        self._federated_training = "FT"
        self._local_training_param = None
        self._metric = Metric(logdir=f"metric/{self._args.name}")

    def do_smpc(self) -> None:
        """The party generates a value among all parties.
        For example: Party A is 0.1; Party B is 0.2; and Party C is -0.3. So when aggregated, the sum value is 0.
        """

        if self._args.peers is None:
            return

        peer_list = self._args.peers.split(";")  # type: ignore
        for index in range(len(peer_list)):
            if index == 0:
                srv = peer_server.PeerServer(len(peer_list) - 1)
                t = Thread(
                    target=base_server.start_server, args=(peer_list[index], srv)
                )
                t.start()
            else:
                cli = peer_client.PeerClient(
                    peer_list[index], self._party_name, self._args.peer_cert
                )
                public_key = cli.get_DH_public_key()
                secret = diffie_hellman_inst.DiffieHellmanInst().generate_secret(
                    public_key
                )
                logger.info(f"secret: {secret}, type: {type(secret)}")

                random_value = cli.get_SMPC_random_key(secret)
                self._sum_random_value += random_value
                logger.info(f"random value: {random_value}")

        self._sum_random_value += srv.sum_parties_random_value()
        logger.info(f"sum all random values: {self._sum_random_value}")

    def exit(self) -> None:
        """Before exiting, the client needs to save the metrics and notify the server of the client's status."""

        self._network_client.transport(message_type.MSG_COMPLETE)
        os._exit(0)

    def run(self) -> None:
        """start training"""

        logger.info("register to server")
        sample_num = self._trainer.config().get("sample_num", 0)
        batch_num = self._trainer.config().get("batch_num", 0)

        while True:
            try:
                resp = self._network_client.transport(
                    message_type.MSG_REGISTER,
                    homo_pb2.RegistrationInfo(
                        sample_num=sample_num, step_num=batch_num
                    ),
                )
                break
            except Exception as e:
                logger.info(e)
                time.sleep(3)

        logger.info(f"use strategy: {resp.strategy}")
        # if resp.parameters:
        #     data_m = dict()
        #     for k, v in resp.parameters.items():
        #         data_m[k] = homo_pb2.Parameter(shape=v.shape)
        #         data_m[k].values.extend(v.values)
        #     self._global_params = {}  # type: ignore
        #     for k, v in data_m.items():
        #         self._global_params[k] = np.asarray(v.values).reshape(v.shape)
        #     self._trainer.set(self._global_params)
        #     logger.info(f"load global model.")

        self.do_smpc()

        if resp.strategy == message_type.STRATEGY_FEDAVG:
            self._strategy = fedavg_client.FedavgClient()
        elif resp.strategy == message_type.STRATEGY_SCAFFOLD:
            self._strategy = fedavg_client.FedavgClient(True)

        elif resp.strategy == message_type.STRATEGY_FEDOPT:
            self._strategy = fedopt_client.FedoptClient()  # type: ignore

        elif resp.strategy == message_type.STRATEGY_qFEDAVG:
            self._strategy = qfedavg_client.qFedavgClient()  # type: ignore

        elif resp.strategy == message_type.STRATEGY_FEDNOVA:
            self._strategy = fednova_client.FedNovaClient()  # type: ignore

        self._network_client.set_strategy(self._strategy)

        t = Thread(target=self._network_client.notice)
        t.start()

        logger.info("report client ready")
        self._network_client.transport(message_type.MSG_CLIENT_READY, None)

        learning_type = self._federated_training
        current_epoch = 0
        while True:
            if (
                self._strategy.current_stage() == StrategyClient.Stage.Training
                or learning_type == self._local_training
            ):
                logger.info(f"----- fit <{learning_type}> -----")
                if learning_type == self._local_training:
                    self._trainer.set(self._local_training_param)  # type: ignore
                    current_epoch = self._epoch - 1
                else:
                    current_epoch = self._epoch

                try:
                    self._strategy.set_trainer_config(self._trainer.config())
                    fit = self._trainer.fit(current_epoch)
                    if isinstance(fit, types.GeneratorType):
                        param = next(fit)
                        while True:
                            param = self._strategy.update_param(param)
                            param = fit.send(param)
                except StopIteration:
                    logger.info("epoch end")

                logger.info(f"----- evaluate <{learning_type}> -----")
                metrics = self._trainer.evaluate(current_epoch)
                if metrics is not None:
                    for k, v in metrics.items():
                        self._metric.add(k, learning_type, current_epoch, v)
                        if (
                            learning_type == self._federated_training
                            and self._epoch == 1
                        ):
                            self._metric.add(k, self._local_training, current_epoch, v)

                logger.info(f"----- get <{learning_type}> -----")
                client_param = self._trainer.get()
                if self._args.enable_ll:
                    if learning_type == self._federated_training:
                        if self._local_training_param is None:
                            self._local_training_param = client_param  # type: ignore
                        else:
                            learning_type = self._local_training
                    else:
                        learning_type = self._federated_training
                        self._local_training_param = client_param  # type: ignore

                        if self._epoch == self._args.epochs:
                            self.exit()

                        continue

                upload_param = self._strategy.generate_upload_param(
                    self._epoch, client_param, metrics
                )
                if self._sum_random_value != 0.0:
                    smpc_data = dict()
                    for k, v in upload_param.parameters.items():
                        smpc_data[k] = homo_pb2.Parameter(shape=v.shape)  # type: ignore
                        smpc_data[k].values.extend(
                            [item + self._sum_random_value for item in v.values]  # type: ignore
                        )
                    upload_param = homo_pb2.UploadParam(
                        epoch=upload_param.epoch, parameters=smpc_data, metrics=metrics
                    )

                if self._epoch == self._args.epochs:
                    if self._args.enable_ll:
                        continue
                    else:
                        self.exit()

                self._network_client.transport(
                    message_type.MSG_UPLOAD_PARAM, upload_param
                )
                self._strategy.set_current_stage(StrategyClient.Stage.Waiting)
                self._epoch += 1
            elif self._strategy.current_stage() == StrategyClient.Stage.Setting:
                logger.info("----- set -----")
                self._global_params = self._strategy.aggregate_result()
                self._trainer.set(self._global_params)
                self._strategy.set_current_stage(StrategyClient.Stage.Waiting)
                self._network_client.transport(message_type.MSG_CLIENT_READY, None)
            else:
                time.sleep(1)
