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
"""Define the heartbeat interval between client and server."""
MSG_HEARTBEAT_INTERVAL = 1

"""Define the message type for communication. (From client to server)"""
MSG_REGISTER = "msg_register"
MSG_CLIENT_READY = "msg_client_ready"
MSG_UPLOAD_PARAM = "msg_upload_param"
MSG_COMPLETE = "msg_complete"

"""Define the message type for communication. (From server to client)"""
MSG_AGGREGATE_RESULT = "msg_aggregate_result"
MSG_NOTIFY_TRAINING = "msg_notify_training"

"""Define the name of strategy."""
STRATEGY_FEDAVG = "FedAvg"
STRATEGY_SCAFFOLD = "Scaffold"
STRATEGY_FEDOPT = "FedOpt"
STRATEGY_qFEDAVG = "qFedAvg"
STRATEGY_FEDNOVA = "FedNova"
