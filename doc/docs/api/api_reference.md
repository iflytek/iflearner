## API Reference

### Trainer

#### 1. class `iflearner.business.homo.trainer.Trainer`

This is the base client which would to be inherited when you implement your own client, and it has four abstract functions containing `get`, `set`, `fit` and `evaluate`.
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

This class inherit from `iflearner.business.homo.trainer.Trainer` and implement two of four functions which are `get` and `set`. 
When you are using PyTorch framework, you can choose to inherit from this class, not the `iflearner.business.homo.trainer.Trainer`. 
Then, you just need to implement `fit` and `evaluate` functions.

#### 3. class `iflearner.business.homo.tensorflow_trainer.TensorFlowTrainer`

This class inherit from `iflearner.business.homo.trainer.Trainer` and implement two of four functions which are `get` and `set`. 
When you are using TensorFlow framework, you can choose to inherit from this class, not the `iflearner.business.homo.trainer.Trainer`. 
Then, you just need to implement `fit` and `evaluate` functions.

#### 4. class `iflearner.business.homo.mxnet_trainer.MxnetTrainer`

This class inherit from `iflearner.business.homo.trainer.Trainer` and implement two of four functions which are `get` and `set`. 
When you are using Mxnet framework, you can choose to inherit from this class, not the `iflearner.business.homo.trainer.Trainer`. 
Then, you just need to implement `fit` and `evaluate` functions.

#### 5. class `iflearner.business.homo.keras_trainer.KerasTrainer`

This class inherit from `iflearner.business.homo.trainer.Trainer` and implement two of four functions which are `get` and `set`. 
When you are using Keras framework, you can choose to inherit from this class, not the `iflearner.business.homo.trainer.Trainer`. 
Then, you just need to implement `fit` and `evaluate` functions.


### Command Arguments

`iflearner.business.homo.argument.parser`

We provide some command arguments in advance, and you need call `parser.parse_args` function at the begining of your program. 
Of course, you can add your own command arguments by calling `parser.add_argument` function before parsering command arguments.

There are default arguments as the follow:

  - name

    You can specify the name of client, and the name need to be unique which can't be the same as other clients.
  
  - epochs

    You can specify the total epochs of training, and when training achieved, the client will exit automatically.

  - server

    You can specify the server address, eg: "192.168.0.1:50001".

  - enable-ll

    We provide local training way at the same time which just use your own data, so you can compare the results between federal training with local training. 
    The argument value is 1 (enable) or 0 (disable, default).

  - peers

    We provide SMPC for secure aggregation, and you can specify addresses to enable this feature.

    eg: '192.168.0.1:50010;192.168.0.2:50010;192.168.0.3:50010'

    First one address is your own address, and other addresses behind are other clients' addresses. All of addresses use semicolon to separate.

### Controller

class `iflearner.business.homo.train_client.Controller`

This class is the driver of client and control the whole process, so you would instantiate the class to start your client.
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

