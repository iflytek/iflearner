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

import os
import pickle
import time
from os.path import join
from typing import Any
from loguru import logger

from iflearner.business.hetero.parser import Parser
from iflearner.business.hetero.builder.builders import Builders
from iflearner.communication.hetero.hetero_network import HeteroNetwork

parser = Parser()

class Driver:
    """Drive the entire process according to the flow yaml.
    
    Flow yaml format:
        role: string
        steps:
        - name: string
          upstreams:
          - role: string
            step: string
    """
    
    def __init__(self) -> None:
        """Init the class.
        """
        logger.add(f"log/{parser.model_name}_{parser.role_name}.log", backtrace=True, diagnose=True)
        if parser.model_name not in Builders:
            raise Exception(f"{parser.model_name} is not existed.")
        
        self._model = Builders[parser.model_name].create_role_model_instance(parser.role_name)
        self._model.set_hyper_params(parser.hyper_params)
    
        parser.parse_model_flow_file(join("model", parser.model_name, Builders[parser.model_name].get_role_model_flow_file(parser.role_name)))    
        logger.info(f"Model flow: {parser.model_flow}")
        logger.info(f"Network config: {parser.network_config}")
        self._network = HeteroNetwork(*parser.network_config)
        self._init_steps = False
    
    def _exec_steps(self, steps: Any) -> None:
        """Execute steps.
        
        Args:
            steps (Any): Details of the steps.
        """
        for step in steps:
            logger.info(f"{step}")
            for upstream in step.upstreams:
                data_list = None
                while data_list is None:
                    data_list = self._network.pull(upstream.role, upstream.step)
                    time.sleep(1)
                    
                self._model.handle_upstream(upstream.role, upstream.step, data_list)
            
            if step.virtual is True:
                continue
            
            result = self._model.handle_step(step.name)
            if result is None:
                continue
            
            for name, data in result.items():
                data = pickle.dumps(data)
                self._network.push(name, step.name, data)
                
    def _exec_model_flow(self) -> None:
        """Execute model flow.
        
        Raise:
            Exception(f"The return type of {step.name} is illegal.")
        """
        if not self._init_steps and parser.model_flow.init_steps is not None:
            self._exec_steps(parser.model_flow.init_steps)
            self._init_steps = True
            
        self._exec_steps(parser.model_flow.steps)
    
    def run(self, epoch: int=1) -> None:
        """Loop execution process.

        Args:
            epoch (int, optional): The number of epochs we need to run. Defaults to 1.
        """
        
        if epoch == -1:
            while True:
                logger.info("Infinite loop")
                self._exec_model_flow()
                
        for i in range(epoch):
            logger.info(f"Start epoch {i+1}")
            self._exec_model_flow()
    
if __name__ == "__main__":
    parser.parse_task_configuration_file()
    driver = Driver()
    driver.run(parser.epochs)
    os._exit(0)
    