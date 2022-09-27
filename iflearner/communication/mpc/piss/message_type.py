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
"""Define the message type for communication. (From client to server)"""
MSG_REGISTER = "msg_register"
MSG_START_QUERY = "msg_start_query"
MSG_PARTICIPANTS_READY = "msg_participants_ready"

"""Define the message type for communication. (From server to client)"""
MSG_PARTICIPANTS_ROUTES = "msg_participants_routes"

"""Define the message type for communication. (Between clients)"""
MSG_SHARE_ENCRYPTION_PARAM = "msg_share_encryption_parameters"
MSG_SHARE_ENCRYPTED_SECRETS  = "msg_share_encrypted_secrets"
MSG_RETURN_ENCRYPTED_DATA_SUM = "msg_return_encrypted_data_sum"

MSG_INIT_DATA = "msg_init_data"


