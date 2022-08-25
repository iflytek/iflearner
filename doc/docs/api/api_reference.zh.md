## API Reference

### Trainer

#### 1. class `iflearner.business.homo.trainer.Trainer`

这是您实现自己的客户端时要继承的基本类，它有四个抽象函数，其中包含 `get`, `set`, `fit` 和 `evaluate`.

```python
def get(self, param_type=ParameterType.ParameterModel) -> dict:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

  In this function, you would implement how to get the parameters or gradients from your model, and we call this function to get your model information. 
  Then, we will send it to server for aggregating with other clients.

  IN: 
    param_type: which one you want to get, parameter or gradient
        - ParameterType.ParameterModel (default)
        - ParameterType.ParameterGradient

  OUT: 
    dict: k: str (the parameter name), v: np.ndarray (the parameter value)

def set(self, parameters: dict, param_type=ParameterType.ParameterModel) -> None:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

  In this function, you would implement how to set the parameters or gradients to your model, and we will call this function when we got aggregating result from server.

  IN: 
    dict: the same as the return of ``get`` function

    param_type: which one you want to set, parameter or gradient
        - ParameterType.ParameterModel (default)
        - ParameterType.ParameterGradient

  OUT: 
    none

def fit(self, epoch: int) -> None:
""""""""""""""""""""""""""""""""""

  In this function, you would implement the training process on one epoch, and we will call this function per epoch.

  IN:
    epoch: the index of epoch

  OUT:
    none

def evaluate(self, epoch: int) -> dict:
"""""""""""""""""""""""""""""""""""""""

  In this function, you would implement the evaluating process for model and return the metrics. We will call this function after called ``fit`` function at every time.

  IN:
    epoch: the index of epoch

  OUT:
    dict: k: str (metric name), v: float (metric value)
```

#### 2. class `iflearner.business.homo.pytorch_trainer.PyTorchTrainer`

这个类继承自 `iflearner.business.homo.trainer.Trainer` 并实现了 `get` 和 `set` 这四个函数中的两个。
当你使用 PyTorch 框架时，你可以选择继承这个类，而不是 `iflearner.business.homo.trainer.Trainer`。
然后，您只需要实现 `fit` 和 `evaluate` 函数。

#### 3. class `iflearner.business.homo.tensorflow_trainer.TensorFlowTrainer`

这个类继承自 `iflearner.business.homo.trainer.Trainer` 并实现了 `get` 和 `set` 这四个函数中的两个。
当你使用 Tensorflow 框架时，你可以选择继承这个类，而不是 `iflearner.business.homo.trainer.Trainer`。
然后，您只需要实现 `fit` 和 `evaluate` 函数。

#### 4. class `iflearner.business.homo.mxnet_trainer.MxnetTrainer`

这个类继承自 `iflearner.business.homo.trainer.Trainer` 并实现了 `get` 和 `set` 这四个函数中的两个。
当你使用 Mxnet 框架时，你可以选择继承这个类，而不是 `iflearner.business.homo.trainer.Trainer`。
然后，您只需要实现 `fit` 和 `evaluate` 函数。

#### 5. class `iflearner.business.homo.keras_trainer.KerasTrainer`

这个类继承自 `iflearner.business.homo.trainer.Trainer` 并实现了 `get` 和 `set` 这四个函数中的两个。
当你使用 Keras 框架时，你可以选择继承这个类，而不是 `iflearner.business.homo.trainer.Trainer`。
然后，您只需要实现 `fit` 和 `evaluate` 函数。


### Command Arguments

`iflearner.business.homo.argument.parser`

我们预先提供了一些命令参数，您需要在程序开始时调用 `parser.parse_args` 函数。
当然，您可以通过在解析命令参数之前调用 `parser.add_argument` 函数来添加自己的命令参数。

有如下默认参数：

  - name

    您可以指定客户端的名称，名称必须是唯一的，不能与其他客户端相同。
  
  - epochs

    您可以指定训练的总轮数，当训练完成时，客户端将自动退出。

  - server

    您可以指定服务端链接地址，例如："192.168.0.1:50001"。

  - enable-ll

    我们同时提供本地培训方式，只使用您自己的数据，因此您可以比较联邦培训与本地培训的结果。 参数值为 1（启用）或 0（禁用，默认值）。

  - peers

    我们为安全聚合提供 SMPC，您可以指定地址来启用此功能。
    例如：'192.168.0.1:50010;192.168.0.2:50010;192.168.0.3:50010'.
    第一个地址是您自己的地址，后面的其他地址是其他客户的地址。 所有地址都使用分号分隔。

### Controller

class `iflearner.business.homo.train_client.Controller`

该类是客户端的驱动程序并控制整个过程，因此您将实例化该类以启动您的客户端。
```python
def __init__(self, args, trainer: Trainer) -> None:
"""""""""""""""""""""""""""""""""""""""""""""""""""

  class initialization function

  IN:

    args: the return of ``iflearner.business.homo.argument.parser.parse_args``

    trainer: the instance of ``iflearner.business.homo.trainer.Trainer``

  OUT:

    none

def run(self) -> None:
""""""""""""""""""""""

  You would call this function after your client has ready, and this function will block until training process has been completed.
  
```

