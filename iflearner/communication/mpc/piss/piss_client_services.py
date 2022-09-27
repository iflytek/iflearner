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
from operator import imod
import timeit
from tkinter import NO
from typing import Any

from loguru import logger
import threading

from iflearner.communication.base import base_pb2
from iflearner.communication.mpc.piss import piss_pb2
from iflearner.communication.mpc.piss import message_type
from iflearner.business.mpc.piss import piss_strategy_client
from iflearner.communication.mpc.piss.piss_exception import PissException
from iflearner.communication.mpc.piss.piss_base import PissBase

class PissClientServices(PissBase):
    """Implement private information statistical summation server base on base_server.BaseServer."""

    def __init__(self, server_addr: str, party_name: str, 
                        route: str, 
                        data_path:str,
                        cert_path: str = None
                        ) -> None:
        super().__init__(party_name, route, cert_path)

        self._secrets_sum = {}
        self._strategy = piss_strategy_client.PissStrategyClient(self._cert_path, self._party_name, self._options, data_path)
        self._server_stub = self._strategy.generate_stub(server_addr)

    def transport(self, type: str, stub: Any = None, data: Any = None)-> None:
        """Transport data betwees client and server or client."""
        try:
            start = timeit.default_timer()
            req = base_pb2.BaseRequest(party_name = self._party_name,
                                        type = type)
            if data is not None:
                req.data = data.SerializeToString()
            resp = None 
            if type == message_type.MSG_REGISTER:
                resp = self._post(self._server_stub, req)
            elif type == message_type.MSG_START_QUERY:
                resp = self._send(self._server_stub, req)
            elif type == message_type.MSG_SHARE_ENCRYPTION_PARAM:
                resp = self._send(stub , req)
            elif type == message_type.MSG_SHARE_ENCRYPTED_SECRETS:
                resp = self._send(stub,req)
            elif type == message_type.MSG_RETURN_ENCRYPTED_DATA_SUM:
                resp = self._post(stub,req)
            if resp.code != 0:  # type: ignore
                raise PissException(code=PissException.PissResponseCode(resp.code), message=resp.message  # type: ignore
                    )
        except PissException as e:
            logger.info(str(e))
            #return base_pb2.BaseResponse(code=e.code, message=e.message)
        finally:
            stop = timeit.default_timer()
            logger.info(f"OUT: message type: {type}, time: {1000 * (stop - start)}ms")
        return resp
        
    def send(self,  request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        """Call send function."""
        try:
            start = timeit.default_timer()
            resp_data = None
            code = 0
            if request.type == message_type.MSG_INIT_DATA:
                req_data = piss_pb2.InitData()
                req_data.ParseFromString(request.data)
                self._strategy.init_data(data_path= req_data.data_path)

            if request.type == message_type.MSG_START_QUERY:
                req_data = piss_pb2.ShareEncryptionParam()
                req_data.ParseFromString(request.data)
                self._strategy.set_encryption_param(req_data)
                resp = self.transport(type = message_type.MSG_START_QUERY)
                if resp.code!=0:
                    raise PissException(code=PissException.PissResponseCode(resp.code), 
                                        message=resp.message  # type: ignore
                                                    )
            elif request.type == message_type.MSG_PARTICIPANTS_READY:
                pass 
            elif request.type == message_type.MSG_PARTICIPANTS_ROUTES:
                req_data = piss_pb2.ParticipantsRoutes()
                req_data.ParseFromString(request.data)
                self._strategy.generate_participants_stubs(req_data)   
                if req_data.initiator_party_name == self._party_name:
                    en_data = piss_pb2.ShareEncryptionParam(encryption_param = self._strategy.get_encryption_param(),
                                                            route = self._route)
                    party_name_list = self._strategy.get_party_name_list() 
                    stubs = self._strategy.get_stubs() 

                    for pn in party_name_list:
                        if pn != self._strategy.get_virtual_client():
                            stub = stubs[pn]
                            resp = self.transport(type = message_type.MSG_SHARE_ENCRYPTION_PARAM,
                                                stub = stub,
                                                data = en_data)
                            if resp.code != 0:  # type: ignore
                                raise PissException(code=PissException.PissResponseCode(resp.code), 
                                                    message=resp.message  # type: ignore
                                                    )

            elif request.type == message_type.MSG_SHARE_ENCRYPTION_PARAM:
                self.timer_sub()
                req_data = piss_pb2.ShareEncryptionParam()
                req_data.ParseFromString(request.data)

                encrypt_secrets_resp = self._strategy.encrypt_secrets(request.party_name, req_data)
                if encrypt_secrets_resp != PissException.PissResponseCode.NotInitiator:
                    self_sub_keys = encrypt_secrets_resp[0]
                    self_commitments = encrypt_secrets_resp[1]
                    party_name_list = self._strategy.get_party_name_list() 
                    stubs = self._strategy.get_stubs() 
                    #sync shares to clients
                    for pn in party_name_list:
                        if pn != self._party_name :
                            stub = stubs[pn]
                            sub_keys = self_sub_keys[pn]
                            data = piss_pb2.ShareEncryptedSecrets(sub_keys = sub_keys, 
                                                                  commitments = self_commitments,
                                                                  recv_party_name = pn
                                                                  )

                            resp = self.transport(type = message_type.MSG_SHARE_ENCRYPTED_SECRETS,
                                                  stub = stub,
                                                  data = data)
                            
                            if resp.code != 0:  # type: ignore
                                raise PissException(code=PissException.PissResponseCode(resp.code), 
                                                    message=resp.message  # type: ignore
                                                    )
                else:
                    raise PissException(PissException.PissResponseCode.NotInitiator, 
                                        "NotInitiator")

            elif request.type == message_type.MSG_SHARE_ENCRYPTED_SECRETS:
                req_data = piss_pb2.ShareEncryptedSecrets()
                req_data.ParseFromString(request.data)
                verify_subkey_resp = self._strategy.verify_subkey(request.party_name,
                                                                    req_data.sub_keys,
                                                                    req_data.commitments,
                                                                    req_data.recv_party_name)
                if not verify_subkey_resp :
                    code = verify_subkey_resp
                    raise  PissException(verify_subkey_resp, 
                                        "verify "+request.party_name +" subkey failed")       
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

    def post(self,  request: base_pb2.BaseRequest, context: Any) -> base_pb2.BaseResponse:
        """Call post function."""
        try:
            start = timeit.default_timer()
            resp_data = None
            code = 0
            if request.type == message_type.MSG_RETURN_ENCRYPTED_DATA_SUM:
                req_data = piss_pb2.SubSecretsSUM()
                req_data.ParseFromString(request.data)
                verify_sumkey_resp = self._strategy.verify_sumkey(request.party_name, req_data.sub_keys_sum)
                if not verify_sumkey_resp :
                    code = verify_sumkey_resp
                    raise  PissException(verify_sumkey_resp, 
                                        "verify "+request.party_name +" sumkey failed")   
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

    def callback(self,  request: base_pb2.BaseRequest) -> base_pb2.BaseResponse:
        """Call callback function."""
        resp = None 
        return resp

    def timer_sub(self):
        t = threading.Timer(2,self.timer_sub)
        if self._strategy.is_start_key_sum():
            t.cancel()
            try:
                start = timeit.default_timer()
                resp = self._strategy.sub_key_sum()
                if resp != 'initiator':
                    sub_key_sum = piss_pb2.SubSecretsSUM(sub_keys_sum = resp)
                    initiator_stub = self._strategy.get_initiator_stub()
                    resp = self.transport(type = message_type.MSG_RETURN_ENCRYPTED_DATA_SUM,
                                            stub = initiator_stub,
                                            data = sub_key_sum) 
                    if resp.code != 0:  # type: ignore
                        raise PissException(code=PissException.PissResponseCode(resp.code), 
                                                message=resp.message  # type: ignore
                                                )
                else:
                    self.timer_sum()
            except PissException as e:
                logger.info(e)
            except Exception as e:
                logger.info(e)
            finally:
                stop = timeit.default_timer()
                logger.info(
                    f"IN: party: {self._party_name}, message type: {message_type.MSG_RETURN_ENCRYPTED_DATA_SUM}, time: {1000 * (stop - start)}ms"
                ) 
        t.start()

    def timer_sum(self):
        t = threading.Timer(2,self.timer_sum)
        if self._strategy.is_start_reconstruct():
            t.cancel()
            try:
                start = timeit.default_timer()
                self._secrets_sum = self._strategy.reconstruct_sum_secrets()
                logger.info(
                    f"IN: party: {self._party_name}, secrets_sum: {self._secrets_sum}"
                ) 
                #print(self._secrets_sum)
            except PissException as e:
                logger.info(e)
            except Exception as e:
                logger.info(e)
            finally:
                stop = timeit.default_timer()
                logger.info(
                    f"IN: party: {self._party_name}, message type: {'timer_sum'}, time: {1000 * (stop - start)}ms"
                ) 
        t.start()
 


