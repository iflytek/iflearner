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
import timeit
from typing import Any

from loguru import logger

from iflearner.business.homo.strategy import strategy_server
from iflearner.communication.base import base_pb2, base_server
from iflearner.communication.homo import homo_pb2, message_type
from iflearner.communication.homo.homo_exception import HomoException


class HomoServer(base_server.BaseServer):
    """Implement homogeneous server base on base_server.BaseServer."""

    def __init__(self, strategy: strategy_server.StrategyServer) -> None:
        self._callback_messages: dict = dict()
        self._strategy = strategy

    def send(
        self, request: base_pb2.BaseRequest, context: Any
    ) -> base_pb2.BaseResponse:
        """Handle client requests synchronously."""

        try:
            start = timeit.default_timer()
            resp_data = None
            if request.type == message_type.MSG_REGISTER:
                data = homo_pb2.RegistrationInfo()
                data.ParseFromString(request.data)
                resp_data = self._strategy.handler_register(
                    request.party_name, data.sample_num, data.step_num
                )
            elif request.type == message_type.MSG_CLIENT_READY:
                resp_data = self._strategy.handler_client_ready(request.party_name)  # type: ignore
            elif request.type == message_type.MSG_COMPLETE:
                resp_data = self._strategy.handler_complete(request.party_name)  # type: ignore
            elif request.type in self._strategy.custom_handlers:
                resp_data = self._strategy.custom_handlers[request.type](
                    request.party_name, request.data
                )
        except HomoException as e:
            logger.info(e)
            return base_pb2.BaseResponse(code=e.code, message=e.message)
        except Exception as e:
            logger.info(e)
            return base_pb2.BaseResponse(
                code=HomoException.HomoResponseCode.InternalError, message=str(e)
            )
        else:
            if resp_data is None:
                return base_pb2.BaseResponse()
            return base_pb2.BaseResponse(data=resp_data.SerializeToString())  # type: ignore
        finally:
            stop = timeit.default_timer()
            logger.info(
                f"IN: party: {request.party_name}, message type: {request.type}, time: {1000 * (stop - start)}ms"
            )

    def post(
        self, request: base_pb2.BaseRequest, context: Any
    ) -> base_pb2.BaseResponse:
        """Handle client requests asynchronously."""

        try:
            start = timeit.default_timer()
            resp_data = None
            if request.type == message_type.MSG_UPLOAD_PARAM:
                req_data = homo_pb2.UploadParam()
                req_data.ParseFromString(request.data)
                resp_data = self._strategy.handler_upload_param(
                    request.party_name, req_data
                )  # type: ignore
            elif request.type in self._strategy.custom_handlers:
                resp_data = self._strategy.custom_handlers[request.type](request.data)
        except HomoException as e:
            logger.info(e)
            return base_pb2.BaseResponse(code=e.code, message=e.message)
        except Exception as e:
            logger.info(e)
            return base_pb2.BaseResponse(
                code=HomoException.HomoResponseCode.InternalError, message=str(e)
            )
        else:
            if resp_data is None:
                return base_pb2.BaseResponse()
            return base_pb2.BaseResponse(data=resp_data.SerializeToString())  # type: ignore
        finally:
            stop = timeit.default_timer()
            logger.info(
                f"IN: party: {request.party_name}, message type: {request.type}, time: {1000 * (stop - start)}ms"
            )

    def callback(
        self, request: base_pb2.BaseRequest, context: Any
    ) -> base_pb2.BaseResponse:
        """The channel of pushing message to clients initiatively."""

        start = timeit.default_timer()
        type, resp_data = self._strategy.get_client_notification(request.party_name)
        if type is not None:
            stop = timeit.default_timer()
            logger.info(
                f"OUT: party: {request.party_name}, message type: {type}, time: {1000 * (stop - start)}ms"
            )

        if resp_data is None:
            return base_pb2.BaseResponse(type=type)

        return base_pb2.BaseResponse(type=type, data=resp_data.SerializeToString())

    # def callback(self, request, context):
    #     if request.party_name in self._callback_messages:
    #         type, data = self._callback_messages.pop(request.party_name)
    #         return base_pb2.BaseResponse(type=type, data=data)

    #     return base_pb2.BaseResponse()

    # '''Send notifications to clients.'''
    # def notice(self):
    #     while (True):
    #         party_name, type, data = self._strategy.get_client_notification()
    #         if party_name in self._callback_messages:
    #             time.sleep(message_type.MSG_HEARTBEAT_INTERVAL)
    #             if party_name in self._callback_messages:
    #                 '''Client may be disconnected.'''
    #                 continue

    #         self._callback_messages[party_name] = tuple(type, data)
