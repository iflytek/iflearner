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

from tkinter.messagebox import NO
import numpy as np
from loguru import logger
from typing import Any, List, Dict, Union
from phe import paillier
from torch import threshold
from iflearner.business.hetero.model.role import Role, guest
from iflearner.business.hetero.model.base_model import BaseModel
from iflearner.business.hetero.model.logistic_regression.dataset import get_host_data
import pandas as pd
# from iflearner.business.hetero.model.Xgboost.base_tree import calculate_gini_impurity, majority_vote, feature_split, node_split
from iflearner.business.hetero.model.Xgboost.base_tree import*
from phe import paillier
import uuid
from sklearn import datasets
from keras.utils.np_utils import*
import os


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
                  left_branch=None, right_branch=None, w_left=None, w_right=None):
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



class XGboostHost(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.base_score = 0.5
        self.calucate_gradient()
        self.loss = LogisticLoss()
        self.current_depth = 0
        self.tree_dict = {}
        self.build_tree_node = ["root"]
        self.build_tree_data = [[self.X, self.current_depth]]
        self.tree_dict_all = {}
        self.tree_num = 0
        self._register_another_step(
            guest, "calu_encry_grdient_hess", self.receive_guest_gradient)
        self._register_another_step(
            guest, "guest_build_tree", self.receive_build_tree)

        self._register_own_step("calucate_gini",
                                self.calucate_gini)
        self._register_own_step(
            "set_hyper_params", self.set_hyper_params)
        self._register_own_step(
            "encry_label", self.encry_label)
        self._register_own_step(
            "calu_gain_compare", self.calu_gain_compare)
        self._register_own_step(
            "build_tree", self.build_tree)
    def calucate_gradient(self):
        """
            Calucate_gradient
        Bind:
            step: calucate_gradient
        """
        x_host = datasets.load_iris().data
        feature_list = datasets.load_iris().feature_names
        x_host = pd.DataFrame(x_host, columns=feature_list)
        # y = to_categorical(datasets.load_iris().target, num_classes=3)
        y_true_pred_host = pd.DataFrame(
            {"label": list(to_categorical(datasets.load_iris().target, num_classes=3))})
        self.X = x_host[[x for x in x_host.columns]]
        self.index_all=self.X.index
        self.Y = y_true_pred_host["label"]
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError('X and Y must have the same length!')
        y = np.array(self.Y.to_list())
        self.y_hat = np.full(y.shape, [self.base_score])
        y_gradient = grad(self.y_hat, y)
        y_hess = hess(self.y_hat, y)
        self.y_gradient_copy = y_gradient.copy()
        self.y_gradient = pd.DataFrame({"g": list(y_gradient)})
        self.y_hess_copy = y_hess.copy()
        self.y_hess = pd.DataFrame({"h": list(y_hess)})
        self.f_t = pd.Series(list(np.zeros(y.shape)))

    def calucate_gini(self):
        """Calucate_gini

            Bind:
                step: calucate_gini

        """
        if not self.build_tree_data:
            self.current_depth = 0
            self.tree_dict_all[self.tree_num] =  self.tree_dict
            self.tree_num += 1
            if self.tree_num < self.n_estimators:
                y = np.array(self.Y.to_list())
                self.f_t = np.array(self.f_t.to_list())
                self.y_hat = self.y_hat+self.learning_rate*self.f_t
                y_gradient = grad(self.y_hat, y)
                y_hess = hess(self.y_hat, y)
                self.y_gradient_copy = y_gradient.copy()
                self.y_gradient = pd.DataFrame({"g":list(y_gradient)})
                self.y_hess_copy = y_hess.copy()
                self.y_hess = pd.DataFrame({"h":list(y_hess)})
                self.f_t = pd.Series(list(np.zeros(y.shape)))
                print("我开始建立新的树了")
                self.tree_dict = {}
                self.build_tree_node = ["root"]
                self.build_tree_data = [[self.X, self.current_depth]]
            else:
                print("发出终止信号")
                os._exit(0)
        pop_data = self.build_tree_data.pop(0)
        data = pop_data[0]
        self.index = data.index
        self.current_depth = pop_data[1]
        max_gain = 0
        # 获取样本数和特征数
        n_samples, n_features = data.shape
        print("n_sample", n_samples)
        print("self.min_samples_split", self.min_samples_split)
        print("self.current_depth", self.current_depth)
        print("self.max_depth", self.max_depth)
        if n_samples >= self.min_samples_split and self.current_depth <= self.max_depth:
            print(1)
            # 遍历计算每个特征的基尼不纯度
            for item in [x for x in data.columns]:
                for cut in list(set(data[item])):
                    if (data.loc[data[item] < cut].shape[0] < self.min_child_sample)\
                            | (data.loc[data[item] >= cut].shape[0] < self.min_child_sample):
                        continue
                    G_left = self.y_gradient.loc[data[data[item] < cut].index.tolist(
                    ), "g"].sum()
                    G_right = self.y_gradient.loc[data[data[item] >= cut].index.tolist()
                        , "g"].sum()
                    H_left = self.y_hess.loc[data[data[item]< cut].index.tolist(), "h"].sum()
                    H_right = self.y_hess.loc[data[data[item]
                                                   >= cut].index.tolist(), "h"].sum()
                    gain = G_left**2/(H_left + self.reg_lambda) + \
                        G_right**2/(H_right + self.reg_lambda) - \
                        (G_left + G_right)**2 / \
                        (H_left + H_right + self.reg_lambda)
                    gain = (gain/2 - self.gamma).sum()
                    if gain > max_gain:
                        best_var, best_cut = item, cut
                        max_gain = gain
                        self.G_left_best, self.G_right_best, self.H_left_best, self.H_right_best = G_left, G_right, H_left, H_right
                        self.best_criteria = {
                            "feature_name": best_var, "threshold": best_cut}
                        self.best_sets = {
                            "left_index": data[data[item] < cut].index.tolist(),
                            "right_index": data[data[item] >= cut].index.tolist(),
                        }
            if max_gain==0:
                self.gain_server = None
            else:
                self.gain_server = max_gain
        else:
            self.best_criteria = None
            self.best_sets = None
            self.gain_server = None

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
        self.max_depth = 1
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

    def encry_label(self):
        """encry_label

        Bind:
            step: encry_label

        Returns:
            Dict[Union[Role, str], Any]: Return guest role name and its encrypted data.
        """
        # 获取同态加密的公私钥
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        # 加密整个数据
        gradient_y_copy = []
        hess_y_copy = []
        for index, row in enumerate(self.y_gradient_copy):
            gradient_y_copy.append([self.public_key.encrypt(x) for x in row])
        for index, row in enumerate(self.y_hess_copy):
            hess_y_copy.append([self.public_key.encrypt(x) for x in row])
        gradient_y_copy = np.array(gradient_y_copy)
        hess_y_copy = np.array(hess_y_copy)
        gradient_y_copy = pd.DataFrame(
            {"gradient": list(gradient_y_copy)}, index=self.index_all)
        hess_y_copy = pd.DataFrame(
            {"hess": list(hess_y_copy)}, index=self.index_all)

        return {guest: (gradient_y_copy, hess_y_copy, self.current_depth, self.index)}

    def receive_guest_gradient(self, data: Dict[str, Any]):
        """receive guest gradient data from the guest.

        Bind:
            step: calu_encry_grdient_hess
            role: guest

        Args:
            data (Dict[str, Any]): Guest party name and gradient data.
        """
        self.feature_splite_receive = list(data.values())[0]
        self.guest_gain = 0
        if self.feature_splite_receive is None:
            self.guest_gain = None
        else:
            for key in self.feature_splite_receive:
                for value in self.feature_splite_receive[key]:
                    gain_value=list(value.values())[0]
                    G_left = np.array([self.private_key.decrypt(x) for x in gain_value[0]])
                    H_left = np.array([self.private_key.decrypt(
                        x) for x in gain_value[1]])
                    G_right = np.array([self.private_key.decrypt(
                        x) for x in gain_value[2]])
                    H_right = np.array([self.private_key.decrypt(
                        x) for x in gain_value[3]])
                    gain = G_left**2/(H_left + self.reg_lambda) + \
                        G_right**2/(H_right + self.reg_lambda) - \
                        (G_left + G_right)**2 / \
                        (H_left + H_right + self.reg_lambda)
                    gain = (gain/2 - self.gamma).sum()
                    if gain > self.guest_gain:
                        self.guest_gain = gain
                        self.guest_feature_i = key
                        self.guest_threshold = list(value.keys())[0]
                        self.guest_w_left = G_left/(H_left+self.reg_lambda)
                        self.guest_w_right = -G_right / \
                            (H_right+self.reg_lambda)
        if self.guest_gain==0:
            self.guest_gain=None

    def calu_gain_compare(self):
        """compare gain from host and guest.

        Bind:
            step: calu_gain_compare

        Returns:
            Dict[Union[Role, str], Any]: Return guest role name and compare result.
        """
        print("server", self.gain_server)
        print("client", self.guest_gain)
        # w_left = - self.G_left_best/(self.H_left_best+self.reg_lambda)
        # w_right = -self.G_right_best/(self.H_right_best+self.reg_lambda)
        if self.gain_server is None and self.guest_gain is None:
            print(1)
            # leaf_value = majority_vote(self.y)
            a_name = self.build_tree_node.pop(0)
            a = TreeNode(index="host")
            self.tree_dict[a_name] = a
            return {guest: None}
        elif self.gain_server is not None and self.gain_server > self.guest_gain:
            print("我是server")
            w_left = - self.G_left_best/(self.H_left_best+self.reg_lambda)
            w_right = -self.G_right_best/(self.H_right_best+self.reg_lambda)
            # print("y_hat",self.y_hat)
            # print("f_t",self.f_t)
            
            # self.y_hat = self.y_hat+self.learning_rate*self.f_t
            for i in self.best_sets["left_index"]:
                self.f_t[i] = list(w_left)
            for i in self.best_sets["right_index"]:
                self.f_t[i] = list(w_right)
            # print("y_hat",self.y_hat)
            print("f_t",self.f_t)
            # self.f_t = np.array(self.f_t.to_list())
            # self.y_hat = self.y_hat+self.learning_rate*self.f_t
            left_branch = str(uuid.uuid1())
            right_branch = str(uuid.uuid1())
            a_name = self.build_tree_node.pop(0)
            print(self.best_criteria["feature_name"])
            a = TreeNode(feature_i=self.best_criteria["feature_name"], threshold=self.best_criteria["threshold"],
                         index="host", left_branch=left_branch, right_branch=right_branch, w_left=w_left, w_right=w_right)
            self.tree_dict[a_name] = a
            self.build_tree_node.append(left_branch)
            self.build_tree_node.append(right_branch)
            datax_append_left = self.X.loc[self.best_sets["left_index"]]
            datax_append_right = self.X.loc[self.best_sets["right_index"]]
            # datay_append_left = self.y_true_pred_host.loc[self.best_sets["left_index"]]
            # datay_append_right = self.y_true_pred_host.loc[self.best_sets["right_index"]]
            self.build_tree_data.append(
                [datax_append_left, self.current_depth+1])
            self.build_tree_data.append(
                [datax_append_right, self.current_depth+1])
            return {guest: None}
        else:
            print(3)
            a_name = self.build_tree_node.pop(0)
            return {guest: ["gain", a_name, self.guest_feature_i, self.guest_threshold, self.guest_w_left, self.guest_w_right]}

    def receive_build_tree(self, data: Dict[str, Any]):
        """Save bulid_tree data from the guest.

        Bind:
            step: guest_build_tree
            role: guest

        Args:
            data (Dict[str, Any]): Guest party name and build_data data.
        """
        self.build_data=list(data.values())[0]
    def build_tree(self):
        """build tree model

        Bind:
            step: build_tree
        """
        if self.build_data == 1:
            pass
        else:
            print("我是client")
            a_name = self.build_data[0]
            a = TreeNode(index="guest")
            self.tree_dict[a_name] = a
            self.build_tree_node.append(self.build_data[1])
            self.build_tree_node.append(self.build_data[2])
            datax_append_left = self.X.loc[self.build_data[3]]
            datax_append_right = self.X.loc[self.build_data[4]]
            
            for i in self.build_data[3]:
                self.f_t[i] = list(self.guest_w_left)
            for i in self.build_data[4]:
                self.f_t[i] = list(self.guest_w_right)
            self.build_tree_data.append(
                [datax_append_left, self.current_depth+1])
            self.build_tree_data.append(
                [datax_append_right, self.current_depth+1])
