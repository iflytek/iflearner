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
from ast import Try
from email import message
import timeit
from typing import Any
from loguru import logger
from iflearner.communication.base import base_pb2, base_server
from iflearner.communication.mpc.piss import piss_pb2, piss_pb2_grpc
from iflearner.communication.mpc.piss import message_type
from iflearner.business.mpc.piss import piss_strategy_server
from iflearner.communication.mpc.piss.piss_exception import PissException
from iflearner.communication.mpc.piss.piss_base import PissBase

class PissServer(PissBase):
    """Implement private information statistical summation server base on base_server.BaseServer."""

    def __init__(self, party_name: str, route: str = None, cert_path: str = None) -> None:

        super().__init__(party_name,route, cert_path)

        self._strategy = piss_strategy_server.PissStrategyServer(self._cert_path, self._party_name, self._options)

    def transport(self, type: str, stub: Any = None, data: Any = None)-> None:
        """Transport data from server to clients"""
        try:
            start = timeit.default_timer()
            req = base_pb2.BaseRequest(party_name = self._party_name,
                                        type = type)
            if data is not None:
                req.data = data.SerializeToString()

            resp = None 
            if type == message_type.MSG_PARTICIPANTS_READY:
                resp = self._send(stub, req)

            elif type ==message_type.MSG_PARTICIPANTS_ROUTES:
                resp = self._send(stub, req)

            if resp.code != 0:  # type: ignore
                raise PissException(code=PissException.PissResponseCode(resp.code), message=resp.message  # type: ignore
                    )
        except PissException as e:
            logger.info(e)
        finally:
            stop = timeit.default_timer()
            logger.info(f"OUT: message type: {type}, time: {1000 * (stop - start)}ms")
        return resp

    def send(self,  request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        """Handle client requests synchronously."""
        try:
            start = timeit.default_timer()
            resp_data = None
            code = 0
            if request.type == message_type.MSG_START_QUERY:
                self._strategy.update_client_info(party_name= request.party_name, 
                                                    type= message_type.MSG_START_QUERY)
                party_name_list = self._strategy.get_party_name_list()
                stubs = self._strategy.get_stubs()

                if len(party_name_list) <3:
                    raise PissException(PissException.PissResponseCode.InsufficientParty, 
                                        "Insufficient number of participants")    
                else :
                    ready_num = 0
                    for pn in party_name_list:
                        if pn != request.party_name:
                            resp = self.transport(type= message_type.MSG_PARTICIPANTS_READY,
                                                  stub= stubs[pn])
                            if resp.code == 0:
                                self._strategy.update_client_info(party_name= pn, 
                                        type= message_type.MSG_PARTICIPANTS_READY)
                                ready_num +=1
                    if ready_num <2:
                        raise PissException(PissException.PissResponseCode.InsufficientParty, 
                                            "Insufficient number of participants")
                    else:
                        data = piss_pb2.ParticipantsRoutes(routes = self._strategy.get_ready_routes(),
                                                            initiator_route = self._strategy.get_initiator_route(),
                                                            initiator_party_name = self._strategy.get_initiator_party_name()
                                                            )

                        start_num = 0
                        for rpn in self._strategy.get_ready_party_name_list():
                            if rpn != request.party_name:
                                resp = self.transport(type= message_type.MSG_PARTICIPANTS_ROUTES,
                                                stub= stubs[rpn],
                                                data= data)
                                if resp.code == 0:  # type: ignore
                                    start_num +=1
                                else:
                                    raise PissException(code=PissException.PissResponseCode(resp.code), 
                                                        message=resp.message  # type: ignore
                                                    )
                        if start_num < 2:
                            raise PissException(PissException.PissResponseCode.InsufficientParty, 
                                            "Insufficient number of participants")
                        else:
                            resp = self.transport(type= message_type.MSG_PARTICIPANTS_ROUTES,
                                                stub= self._strategy.get_initiator_stub(),
                                                data= data)

                            if resp.code !=0:  # type: ignore
                                raise PissException(code=PissException.PissResponseCode(resp.code), 
                                                        message=resp.message  # type: ignore
                                                    )
                            
        except PissException as e:
            logger.info(e)
            return base_pb2.BaseResponse(code=e.code, message=e.message)
        except Exception as e:
            logger.info(e)
            return base_pb2.BaseResponse(
                code=PissException.PissResponseCode.InternalError, message=str(e)
                )
        else:
            if resp_data is None:
                return base_pb2.BaseResponse(code = code)
            return base_pb2.BaseResponse(code = code, data=resp_data.SerializeToString())  # type: ignore
        finally:
            stop = timeit.default_timer()
            logger.info(
                f"IN: party: {request.party_name}, message type: {request.type}, time: {1000 * (stop - start)}ms"
            )

    def post(self,  request: base_pb2.BaseRequest ,context: Any) -> base_pb2.BaseResponse:
        """Handle client requests asynchronously."""
        try:
            start = timeit.default_timer()
            resp_data = None
            code = 0
            if request.type == message_type.MSG_REGISTER:
                req_data = piss_pb2.RegistrationInfo()
                req_data.ParseFromString(request.data)
                self._strategy.update_client_info(request.party_name, 
                                                    type= message_type.MSG_REGISTER, 
                                                    route = req_data.route)

        except PissException as e:
            logger.info(e)
            return base_pb2.BaseResponse(code=e.code, message=e.message)
        except Exception as e:
            logger.info(e)
            return base_pb2.BaseResponse(
                code=PissException.PissResponseCode.InternalError, message=str(e)
                )
        else:
            if resp_data is None:
                return base_pb2.BaseResponse(code = code)
            return base_pb2.BaseResponse(code = code, data=resp_data.SerializeToString())  # type: ignore
        finally:
            stop = timeit.default_timer()
            logger.info(
                f"IN: party: {request.party_name}, message type: {request.type}, time: {1000 * (stop - start)}ms"
            )

    def callback(self, request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:

        resp = None
        return resp

