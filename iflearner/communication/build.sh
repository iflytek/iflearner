python -m grpc_tools.protoc --proto_path=base --python_out=base --grpc_python_out=base base.proto
sed -i 's/^import .*_pb2 as/from iflearner.communication.base \0/' base/base_pb2_grpc.py

python -m grpc_tools.protoc --proto_path=homo --python_out=homo --grpc_python_out=homo homo.proto
