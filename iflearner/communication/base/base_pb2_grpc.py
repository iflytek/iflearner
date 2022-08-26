# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from iflearner.communication.base import base_pb2 as base__pb2


class BaseStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.send = channel.unary_unary(
                '/Base/send',
                request_serializer=base__pb2.BaseRequest.SerializeToString,
                response_deserializer=base__pb2.BaseResponse.FromString,
                )
        self.post = channel.unary_unary(
                '/Base/post',
                request_serializer=base__pb2.BaseRequest.SerializeToString,
                response_deserializer=base__pb2.BaseResponse.FromString,
                )
        self.callback = channel.unary_unary(
                '/Base/callback',
                request_serializer=base__pb2.BaseRequest.SerializeToString,
                response_deserializer=base__pb2.BaseResponse.FromString,
                )


class BaseServicer(object):
    """Missing associated documentation comment in .proto file."""

    def send(self, request, context):
        """Use this function to transport information synchronously.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def post(self, request, context):
        """Use this function to transport information asynchronously.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def callback(self, request, context):
        """Use this function to wait for server information.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BaseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'send': grpc.unary_unary_rpc_method_handler(
                    servicer.send,
                    request_deserializer=base__pb2.BaseRequest.FromString,
                    response_serializer=base__pb2.BaseResponse.SerializeToString,
            ),
            'post': grpc.unary_unary_rpc_method_handler(
                    servicer.post,
                    request_deserializer=base__pb2.BaseRequest.FromString,
                    response_serializer=base__pb2.BaseResponse.SerializeToString,
            ),
            'callback': grpc.unary_unary_rpc_method_handler(
                    servicer.callback,
                    request_deserializer=base__pb2.BaseRequest.FromString,
                    response_serializer=base__pb2.BaseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Base', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Base(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def send(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Base/send',
            base__pb2.BaseRequest.SerializeToString,
            base__pb2.BaseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def post(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Base/post',
            base__pb2.BaseRequest.SerializeToString,
            base__pb2.BaseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def callback(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Base/callback',
            base__pb2.BaseRequest.SerializeToString,
            base__pb2.BaseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)