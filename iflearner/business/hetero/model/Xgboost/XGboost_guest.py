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

import numpy as np
from phe import paillier
from loguru import logger
from typing import Any, List, Dict, Union

from torch import threshold
from iflearner.business.hetero.model.role import Role, host, arbiter
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.model.logistic_regression.dataset import get_guest_data
from iflearner.business.hetero.model.Xgboost.base_tree import*
import pandas as pd
import uuid
from sklearn import datasets


class TreeNode():
    """build the tree
    
    Args:
        index:Executive party
        featute_i:feature name()
        threshold:Threshold segmentation
        left_branch:left branch name
        right_branch:right branch name 
        w_left:left weight
        w_right:right weight
    """
    def __init__(self, index=None, feature_i=None, threshold=None,
                 leaf_value=None, left_branch=None, right_branch=None, w_left=None, w_right=None):
        self.index = index
        # 特征索引
        self.feature_i = feature_i
        # 特征划分阈值
        self.threshold = threshold
        # 叶子节点取值
        # self.leaf_value = leaf_value
        # 左子树
        self.left_branch = left_branch
        # 右子树
        self.right_branch = right_branch
        # 左权重
        self.w_left = w_left
        # 右权重
        self.w_right = w_right


class XGboostGuest(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        # self.x,  = get_guest_data()
        self.tree_dict={}
        self.tree_num=0
        self.tree_dict_all = {}
        self.x_guest = datasets.load_iris().data[:, 2:]
        self.feature_list = datasets.load_iris().feature_names[2:]
        self.x_guest=pd.DataFrame(self.x_guest,columns=self.feature_list)
        self._register_another_step(
            host, "encry_label", self.receive_lable_index)
        self._register_another_step(
            host, "calu_gain_compare", self.receive_gain_compare)
        self._register_own_step("set_hyper_params",
                                self.set_hyper_params)
        self._register_own_step(
            "calu_encry_grdient_hess", self.calu_encry_grdient_hess)
        self._register_own_step(
            "guest_build_tree", self.guest_build_tree)

    def set_hyper_params(self, hyper_params: Any) -> None:
        """Set hyper params.

        Args:
            hyper_params (Any): Details of the hyper params.
        """
        # 树的棵树
        self.n_estimators = 3
        # 学习率
        self.learning_rate = 0.001
        # 结点分裂最小样本数
        self.min_samples_split = 2
        # 结点最小基尼不纯度
        self.min_gini_impurity = 999
        # 树最大深度
        self.max_depth = 2
        # 初始化权重
        self.base_score = 0.5
        # L2正则权重系数
        self.reg_lambda = 1
        # 正则项中，叶子节点数的权重系数
        self.gamma = 0
        # 设置最小的分支数量
        self.min_child_sample = 3
        if hyper_params is not None:
            # 树的棵树
            self.n_estimators = hyper_params.get("n_estimators", 300)
            # 学习率
            self.learning_rate = hyper_params.get("n_estimators", 0.001)
            # 结点分裂最小样本数
            self.min_samples_split = hyper_params.get("n_estimators", 2)
            # 结点最小基尼不纯度
            self.min_gini_impurity = hyper_params.get("n_estimators", 999)
            # 树最大深度
            self.max_depth = hyper_params.get("n_estimators", 2)
            # 初始化权重
            self.base_score = hyper_params.get("base_score", 0.5)
            # L2正则权重系数
            self.reg_lambda = hyper_params.get("reg_lambda", 1)
            # 正则项中，叶子节点数的权重系数
            self.gamma = hyper_params.get("gamma", 0)
            # 设置最小的分支数量
            self.min_child_sample = hyper_params.get("min_child_sample", 2)

    def receive_lable_index(self, data: Dict[str, Any]):
        """Save encrypted data from the host.

        Bind:
            step: encry_label
            role: host

        Args:
            data (Dict[str, Any]): Host party name and encrypted data.
        """
        self.y_gradient, self.y_hess, self.current_depth ,self.index= list(data.values())[0]
      
    def calu_encry_grdient_hess(self):
        """calucate gradient adn hess data.

        Bind:
            step: calu_encry_grdient_hess

        Returns:
            Dict[Union[Role, str], Any]: Return host role name and its gradient and hess data.
        """
        if self.tree_dict:
            self.tree_dict[self.tree_num]=self.tree_dict
            self.tree_num+=1
            self.tree_dict={}
        feature_splite_send = {}
        self.feature_splite_data = {}
    
        # 合并输入和标签
        data=self.x_guest.loc[self.index]
        # 获取样本数和特征数
        n_samples, n_features = data.shape
        if n_samples >= self.min_samples_split and self.current_depth <= self.max_depth:
            for feature_i,item in enumerate([x for x in data.columns]):
                feature_splite_send[feature_i]=[]
                self.feature_splite_data[feature_i]=[]
                for cut in list(set(data[item])):
                    if (data.loc[data[item] < cut].shape[0] < self.min_child_sample)\
                            | (data.loc[data[item] >= cut].shape[0] < self.min_child_sample):
                        continue
                    G_left = self.y_gradient.loc[data[data[item] < cut].index.tolist(),"gradient"].sum()
                    G_right = self.y_gradient.loc[data[data[item] >= cut].index.tolist(),"gradient"].sum()
                    H_left = self.y_hess.loc[data[data[item] < cut].index.tolist(),"hess"].sum()
                    H_right = self.y_hess.loc[data[data[item] >= cut].index.tolist(),"hess"].sum()
                    # print("G_left",G_left)
                    # print("cut",cut)
                    # print(len(data[data[item] < cut].index.tolist()))
                    feature_splite_send[feature_i].append({threshold: 
                        (G_left, H_left, G_right, H_right)})
                    best_criteria = {
                        "feature_name": self.feature_list[feature_i], "threshold": threshold}
                    best_sets = {
                        "left_index": data[data[item] < cut].index.tolist(),
                        # "lefty": xy1[:, n_features:],
                        "right_index": data[data[item] >= cut].index.tolist(),
                        # "righty": xy2[:, n_features:],
                    }
                    self.feature_splite_data[feature_i].append(
                        {threshold: [best_criteria, best_sets]})
        else:
            feature_splite_send = None
        return {host: feature_splite_send}

    def receive_gain_compare(self, data: Dict[str, Any]):
        """Save compared data from the host.

        Bind:
            step: calu_gain_compare
            role: host

        Args:
            data (Dict[str, Any]): Host party name and compared data.
        """
        self.gain_compare = list(data.values())[0]

    def guest_build_tree(self):
        """ build tree.

        Bind:
            step: guest_build_tree

        Returns:
            Dict[Union[Role, str], Any]: Return guest role name and its build tree result.
        """
        if self.gain_compare is None:
            print("yilunjies")
            return {host: 1}
        elif self.gain_compare[0] == "gain":
            left_branch = uuid.uuid1()
            right_branch = uuid.uuid1()
            
            feature_i = self.gain_compare[2]
            for dict_threshold in self.feature_splite_data[feature_i]:
                if list(dict_threshold.keys())[0] == self.gain_compare[3]:
                    left_index = list(dict_threshold.values())[0][1]["left_index"]
                    right_index= list(dict_threshold.values())[0][1]["right_index"]
                    # rightx = list(dict_threshold.values())[0][1]["rightx"]
                    # righty = list(dict_threshold.values())[0][1]["righty"]
                    feature_name = list(dict_threshold.values())[
                        0][0]["feature_name"]
                    threshold = list(dict_threshold.values())[
                        0][0]["threshold"]
            a_name = self.gain_compare[1]
            w_left=self.gain_compare[4]
            w_right=self.gain_compare[5]
            a = TreeNode(feature_i=feature_name, threshold=threshold,
                         index="guest", left_branch=left_branch, right_branch=right_branch, w_left=w_left, w_right=w_right)
            self.tree_dict[a_name] = a
            return{host: [a_name, left_branch, right_branch, left_index, right_index]}
        else:
            a = TreeNode(index="host", leaf_value=self.gain_compare[1])

            self.tree_dict[self.gain_compare[2]
                           ]= a
            return {host:None}
