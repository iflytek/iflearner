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

from random import random
from tkinter import NO
from typing import Any, Dict
import numpy as np
import json
from importlib import import_module
from typing import Any, Dict
from loguru import logger

import copy
import pandas as pd
import hashlib
import random
from iflearner.communication.mpc.piss import message_type
from iflearner.communication.mpc.piss.piss_exception import PissException
from iflearner.communication.mpc.piss import piss_pb2
from iflearner.business.mpc.piss.piss_startegy_base import PissStrategyBase 
from iflearner.communication.secureprotol.secretsharing.feldman_verifiable_secretsharing. \
    feldman_verifiable_secret_sharing \
    import FeldmanVerifiableSecretSharing

class PissStrategyClient(PissStrategyBase):
    """Implement the strategy of client.
        Args:
        cert_path (str): 
        party_name (str): the name of party
        options
        data_path (str):the path of .csv data

    Attributes:
        _data_name_dict (Dict[original data_name, Encrypted data_name]) : A dict of original and encrypted data names
        _hash (sha256): For encryption data name
        _meta_data (Dict[Encrypted data_name ,(original data_name, data_values)]):A dict of data values and encrypted data names
        _participants_counts (int):Number of participants
        _vss (FeldmanVerifiableSecretSharing):For secret sharing
        _data_path(str):the path of .csv data
        _encryption_param(Dict[str,str]):Use to store encrypted data name,and share it with other participants
        _list_encryption_param(list): list _encryption_param
        _commitments {Dict[str,str]}:Self generated commitments
        _sub_keys (Dict[str,str]): Self generated sub_keys
        _commitments_recv (Dict[str,str]):commitments received from other parties
        _sub_keys_recv (Dict[str,str]) : sub_keys received from other parties
        _recv_sub_num (int) :sub_key summation counting
        _recv_sum_num (int) :sum_key summation counting
        _virtual_client (str) :A virtual client is used to calculate redundant sub_key
        _party_name_list (str) :
    """
    

    def __init__(self ,cert_path: str, party_name: str,options , data_path:str) -> None:

        super().__init__(cert_path, party_name, options)

        self._data_name_dict = {}
        self._hash = hashlib.sha256()
        self._meta_data = {}
        self._participants_counts = 0
        self._vss = FeldmanVerifiableSecretSharing()

        self._data_path = data_path
        if self._data_path is not None:
            self.init_data(self._data_path)
        
        self._encryption_param = {}
        self._list_encryption_param = []

        self._commitments = {}
        self._sub_keys = {}
        self._commitments_recv = {}
        self._sub_keys_recv = {}

        self._virtual_commitments_recv = {}
        self._virtual_sub_keys_recv = {}
        self._virtual_sub_keys_sum = {}

        self._sub_keys_sum = {}
        self._sub_keys_sum_recv = {}

        self._recv_sub_num = 0
        self._recv_sum_num = 0

        self._virtual_client = 'virtual_client'
        self._party_name_list = [self._virtual_client]
        self._secrets_sum_ready = False

    def get_party_name_list(self):
        """
        Returns:
            self._party_name_list
        """
        return self._party_name_list

    def get_virtual_client(self):
        """
        Returns:
            self._virtual_client name
        """
        return  self._virtual_client

    def get_stubs(self):
        """
        Returns:
            self._stubs:Connection handles for all parties.
        """
        return self._stubs

    def get_initiator_stub(self):
        """
        Returns:
            self._initiator_stub:Connection handles for initiator.
        """
        return self._initiator_stub

    def get_initiator_party_name(self):
        """
        Returns:
            self._initiator_party_name.
        """
        return self._initiator_party_name

    def generate_participants_stubs(self, data: piss_pb2.ParticipantsRoutes)-> None:
        """
        Generate handles for all parties.
        """
        self._initiator_party_name = data.initiator_party_name
        self._initiator_route = data.initiator_route
        self._initiator_stub = self.generate_stub(data.initiator_route)

        self._stubs[self._virtual_client] = self._initiator_stub

        self._routes = data.routes
        self._participants_counts = len(data.routes)
        self._vss.set_share_amount(self._participants_counts)
        list_routes = list(self._routes.items())
        for i in  list_routes:
            self._party_name_list.append(i[0])
            self._stubs[i[0]] = self.generate_stub(i[1])

    def set_encryption_param(self, data):
        """
        The initiator sets encryption parameters and some confusing parameters
        """
        data_index = list(data.encryption_param.items())
        data_name_list = list(self._data_name_dict.keys())

        j = []
        for i in range(len(data_index)*4):
            s = self.random_str()
            if s not in j:
                j.append(s)

        n = 0
        t_data_name = []
        for i in data_index:
            data_name = i[0]+':'+i[1]
            t_data_name.append(data_name)
            self._encryption_param[str(j[n])] = self._data_name_dict[data_name]
            n += 1

        for i in range(len(data_index)):
            r_data_name = data_name_list[random.randint(0, len(data_name_list))]
            if r_data_name not in  t_data_name:
                self._encryption_param[str(j[n])] = self._data_name_dict[r_data_name]
                n += 1

            r_name = self.random_str()
            self._hash.update(r_name.encode("utf-8"))
            self._encryption_param[str(j[n])] = self._hash.hexdigest()
            n += 1
        self._encryption_param = self.random_dict(self._encryption_param)

    def random_dict(self,dicts):
        dict_key_ls = list(dicts.keys())
        random.shuffle(dict_key_ls)
        new_dict = {}
        for key in dict_key_ls:
            new_dict[key] = dicts.get(key)
        return new_dict
    
    def random_str(self, random_length = 10):
        str = ''
        chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789!@#$%^&*()~!'
        length = len(chars) - 1
        for i in range(random_length):
            str += chars[random.randint(0, length)]
        return str

    def get_encryption_param(self):
        """
        Returns
        self._encryption_param(Dict[str,str]):
        Use to store encrypted data name,and share it with other participants"""
        return self._encryption_param

    def encrypt_secrets(self, party_name, data: piss_pb2.ShareEncryptionParam):
        """
        generated commitments and sub_keys
        """
        self._party_name_list.sort()
        if self._initiator_party_name == party_name and self._initiator_route == data.route:
            self._list_encryption_param = list(data.encryption_param.items())
            self._list_encryption_param.sort(key = lambda e:e[1])
            self._vss.key_pair()

            metadata = self.get_data(self._list_encryption_param)
            encrypt_result = self.generate_shares(values = metadata)
            sub_key_table = encrypt_result[0]
            commitments = encrypt_result[1]

            #generate self._sub_keys
            for i in range(self._participants_counts +1):
                sub_key_dict = {}
                sub_key = sub_key_table[:,i]
                s_i=0
                for j in sub_key:
                    sub_key_dict[self._list_encryption_param[s_i][1]] = str(j[1])
                    s_i += 1
                    self._sub_keys[self._party_name_list[i]] = sub_key_dict

            #generate self._commitments
            c_i = 0
            for i in commitments:
                commitment_dict = {}
                for j in range(len(self._party_name_list)):
                    commitment_dict[j+1] = str(i[j])
                self._commitments[self._list_encryption_param[c_i][1]] = json.dumps(commitment_dict)
                c_i +=1

            self._sub_keys_recv[self._party_name] = self._sub_keys[self._party_name]
            self._commitments_recv[self._party_name] = self._commitments
            self._recv_sub_num += 1

            return (self._sub_keys, self._commitments) 
        else :
            return PissException.PissResponseCode.NotInitiator 

    def generate_shares(self,values):
        keys = []
        commitments = []
        for s in values:
            sub_key, commitment = self._vss.encrypt(s)
            keys.append(sub_key)
            commitments.append(commitment)
        res = (np.array(keys), np.array(commitments))
        return res

    def is_start_key_sum(self):
        """
        Used to judge whether to start sub_key summation
        """

        if self._party_name == self._initiator_party_name:
            if self._recv_sub_num == ((len(self._party_name_list)-1)*2):
                return True
            else:
                return False
        else:
            if self._recv_sub_num == (len(self._party_name_list)-1):
                return True
            else:
                return False

    def c_sub_keys(self,keys_recv):
        """
         sub key Sum
        """
        n = None
        for client in  self._party_name_list:
            list_recv = list(keys_recv[client].items())
            """sort by data index"""
            list_recv.sort(key = lambda c:c[0])
            l = list(map(lambda x:int(x[1]), list_recv))
            if n is None:
                n = np.array([l])
            else:
                n = np.concatenate((n,[l]),axis = 0)
        sub_key_sum_values = n.sum(axis=0)
        sub_keys_sum = {}

        j = 0
        for i in sub_key_sum_values:
            sub_keys_sum[self._list_encryption_param[j][1]] = str(i)
            j +=1
        return sub_keys_sum

    def sub_key_sum(self):
        if self._virtual_client in self._party_name_list:
            self._party_name_list.remove(self._virtual_client)
            self._party_name_list.sort()

        if self._party_name == self._initiator_party_name:
            self._sub_keys_sum = self.c_sub_keys(self._sub_keys_recv)
            self._virtual_sub_keys_sum = self.c_sub_keys(self._virtual_sub_keys_recv)
            self._sub_keys_sum_recv[self._party_name] = self._sub_keys_sum
            self._sub_keys_sum_recv[self._virtual_client] = self._virtual_sub_keys_sum
            self._recv_sum_num += 1
            return message_type.MSG_INITIATOR
        else:                
            self._sub_keys_sum = self.c_sub_keys(self._sub_keys_recv)
            return self._sub_keys_sum 

    def verify_subkey(self, send_party_name, recv_sub_keys, recv_commitments,recv_party_name) -> bool:
        """
        Verify the sub keys received from other parties
        Returns: bool
        """
        result = self.verify(send_party_name , recv_sub_keys, recv_commitments,'sub_key')
        if result:
            if recv_party_name == self._virtual_client:
                self._virtual_sub_keys_recv[send_party_name] = recv_sub_keys
                self._virtual_commitments_recv[send_party_name] = recv_commitments
                self._recv_sub_num += 1
            else:
                self._sub_keys_recv[send_party_name] = recv_sub_keys
                self._commitments_recv[send_party_name] = recv_commitments
                self._recv_sub_num +=1
            return True
        else :
            return result

    def verify_sumkey(self, send_party_name, recv_sum_keys):
        """
        The initiator is used to verify the sum keys received from other parties
        Returns: bool
        """
        result = self.verify(send_party_name, recv_sum_keys, None,'sum_key')
        if result:
            self._sub_keys_sum_recv[send_party_name] = recv_sum_keys
            self._recv_sum_num +=1
            return True
        else:
            return result

    def is_start_reconstruct(self):
        if self._recv_sum_num == len(self._party_name_list):
            return True
        else:
            return False

    def secrets_sum_ready(self):
        return self._secrets_sum_ready

    def reconstruct_sum_secrets(self):
        """
        Reconstruct secret
        """
        x = []
        y = []
        if self._virtual_client not in self._party_name_list:
            self._party_name_list.append(self._virtual_client)
            self._party_name_list.sort()

        for client in  self._party_name_list:
            list_recv = list(self._sub_keys_sum_recv[client].items())
            """sort by data index"""
            list_recv.sort(key = lambda c:c[0])
            l = list(map(lambda x:int(x[1]), list_recv))
            x.append(self._party_name_list.index(client)+1)
            y.append(l)
        secrets_sum = self.decrypt(x,y)
        #print(secrets_sum)
        self._secrets_sum_ready = True
        return secrets_sum

    def decrypt(self,x,y):
        secret_sum = []
        for i in range(len(self._list_encryption_param)):
            if self._list_encryption_param[i][1] in self._meta_data:
                y_s = list(map(lambda s:int(s[i]), y))
                s = self._vss.decrypt(x, y_s)
                secret_sum.append((self._meta_data[self._list_encryption_param[i][1]][0],s))
        return secret_sum
        
    def verify(self, send_party_name = None, recv_keys= None, recv_commitments= None , key_type = None):
        if key_type == 'sub_key':
            for i in self._list_encryption_param:
                sub_key =( self._party_name_list.index(send_party_name)+1,int(recv_keys[i[1]]))
                commitment_r = list(json.loads(recv_commitments[i[1]]).items())
                commitment_r.sort(key = lambda e:e[0])
                commitment = list(map(lambda c:int(c[1]), commitment_r))
                res = self._vss.verify(sub_key,commitment)
                if not res:
                    return PissException.PissResponseCode.VerifySubkeyFailed

        elif key_type == 'sum_key':
            party_name_list = copy.deepcopy(self._party_name_list)
            if self._virtual_client not in party_name_list:
                party_name_list.append(self._virtual_client)
                party_name_list.sort()

            for i in self._list_encryption_param:
                sum_key =( party_name_list.index(send_party_name)+1,int(recv_keys[i[1]]))
                commitment_r = list(json.loads(self._commitments_recv[send_party_name][i[1]]).items())
                commitment_r.sort(key = lambda e:e[0])
                commitment_r = list(map(lambda c:int(c[1]), commitment_r))
                commitment_s = list(json.loads(self._commitments[i[1]]).items())
                commitment_s.sort(key = lambda e:e[0])
                commitment_s = list(map(lambda c:int(c[1]), commitment_s))
                commitment = (np.array(commitment_s) * np.array(commitment_r)) % self._vss.p
                res = self._vss.verify(sum_key,commitment)
                if not res:
                    return PissException.PissResponseCode.VerifySumkeyFailed
        return True

    def init_data(self,data_path: str):
        data_all =  pd.read_csv(data_path,encoding='utf-8')
        data_name_list = list(data_all)
        for n in data_name_list:
            if n == data_name_list[0]:
                continue
            else:
                for tup in zip(data_all[data_name_list[0]], data_all[n]):
                    data_name_str = str(tup[0])+ ':' +str(n)
                    data_name = copy.deepcopy(data_name_str)
                    self._hash.update(data_name.encode("utf-8"))
                    data_name = self._hash.hexdigest()
                    self._meta_data[data_name] = (data_name_str,tup[1])
                    self._data_name_dict[data_name_str] = data_name

    def get_data(self , data_index: list):
        metadata = []
        for i in  data_index:
            if i[1] in self._meta_data:
                metadata.append(self._meta_data[i[1]][1])
            else:
                metadata.append(0)
        return metadata


