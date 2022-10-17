import numpy as np


def feature_split(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        def split_func(sample): return sample[feature_i] >= threshold
    else:
        def split_func(sample): return sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_left, X_right])


def cal_gain( y, y_pred, request="server"):
    loss=LogisticLoss()
    if request == "server":
        Gradient = np.power((y * loss.gradient(y, y_pred)).sum(), 2)
        # Hessian矩阵计算
        Hessian = loss.hess(y, y_pred).sum()
        return 0.5 * (Gradient / Hessian)
    elif request == "client":
        return y *loss.gradient(y, y_pred), loss.hess(y, y_pred)


def node_split( y):
    # 中间特征所在列
    feature = int(len(y[0])/2)
    # 左子树为真实值，右子树为预测值
    y_true, y_pred = y[:, :feature], y[:, feature:]
    return y_true, y_pred


def calculate_gini_impurity( y, y1, y2, request="server"):
    if request == "server":
        y = np.array(y.to_list())
        y1 = np.array(y1["label"].to_list())
        y2 = np.array(y2["label"].to_list())
        y_true, y_pred = node_split(y)
        y1, y1_pred = node_split(y1)
        y2, y2_pred = node_split(y2)
        true_gain = cal_gain(y1, y1_pred, request)
        false_gain = cal_gain(y2, y2_pred, request)
        gain = cal_gain(y_true, y_pred, request)
        return true_gain + false_gain - gain
    else:
        y = np.array(y.to_list())
        y1 = np.array(y1["label"].to_list())
        y2 = np.array(y2["label"].to_list())
        y_true, y_pred = node_split(y)
        y1, y1_pred = node_split(y1)
        y2, y2_pred = node_split(y2)
        true_gain_gradient, true_gain_hess = gain(y1, y1_pred, request)
        false_gain_gradient, false_gain_hess = gain(y2, y2_pred, request)
        gain_gradient, gain_hess = gain(y_true, y_pred, request)
        return true_gain_gradient, true_gain_hess, false_gain_gradient, false_gain_hess, gain_gradient, gain_hess


def majority_vote( y):
    loss=LogisticLoss()
    y_true, y_pred = node_split(y)
    # 梯度计算
    gradient = np.sum(y_true * loss.gradient(y_true, y_pred), axis=0)
    # hessian矩阵计算
    hessian = np.sum(loss.hess(y_true, y_pred), axis=0)
    # 叶子结点得分
    leaf_weight = gradient / hessian
    return leaf_weight


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

# 定义Logit损失


class LogisticLoss:
    def __init__(self):
        sigmoid = Sigmoid()
        self._func = sigmoid
        self._grad = sigmoid.gradient

    # 定义损失函数形式
    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self._func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    # 定义一阶梯度
    def gradient(self, y, y_pred):
        p = self._func(y_pred)
        return -(y - p)

    # 定义二阶梯度
    def hess(self, y, y_pred):
        p = self._func(y_pred)
        return p * (1 - p)


def gradient_hess_sum(gradient_y, hess_y, gradient_y1, hess_y1, gradient_y2, hess_y2):
    gradient_y = np.array(gradient_y["gradient"].to_list())
    hess_y = np.array(hess_y["hess"].to_list())
    gradient_y1 = np.array(gradient_y1["gradient"].to_list())
    hess_y1 = np.array(hess_y1["hess"].to_list())
    gradient_y2 = np.array(gradient_y2["gradient"].to_list())
    hess_y2 = np.array(hess_y2["hess"].to_list())
    return gradient_y.sum(), hess_y.sum(), gradient_y1.sum(), hess_y1.sum(), gradient_y2.sum(), hess_y2.sum()


def grad(y_hat, Y):
        '''
        计算目标函数的一阶导
        '''
        y_hat = 1.0/(1.0+np.exp(-y_hat))
        return y_hat - Y
        # elif self.objective == 'linear':
        #     return y_hat - Y
        # else:
        #     raise KeyError('objective must be linear or logistic!')

def hess( y_hat, Y):
        '''
        计算目标函数的二阶导
        '''
        y_hat = 1.0/(1.0+np.exp(-y_hat))
        return y_hat * (1.0 - y_hat)
        # elif self.objective == 'linear':
        #     return np.array([1]*Y.shape[0])
        # else:
        #     raise KeyError('objective must be linear or logistic!')