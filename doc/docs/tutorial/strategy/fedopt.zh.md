## FedOpt
根据文章[***Adaptive Federated Optimization***](https://arxiv.org/abs/2003.00295)，我们实现了fedopt聚合算法，如fedadam、fedyogi、fedadagrad...

完整的该示例的源代码
参考[FedOpt](https://git.iflytek.com/TURING/iflearner/-/tree/master/examples/homo/FedOpt)。

您可以通过继承 `FedOpt` 来实现自己的 fedopt 聚合算法

```python
class FedOpt:
    """Implementation based on https://arxiv.org/abs/2003.00295."""

    def __init__(
        self,
        params: Dict[str, npt.NDArray[np.float32]],
        learning_rate: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        t: float = 0.001,
    ) -> None:
        self._params = params
        self._lr = learning_rate
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._adaptivity = t

    def step(
        self, pseudo_gradient: Dict[str, npt.NDArray[np.float32]]
    ) -> Dict[str, npt.NDArray[np.float32]]:
        '''Update parameters with optimization algorithm according to pseudo gradient'''
        pass
```

如果要使用 Pytorch 版本的优化器，下面是一个例子。 您可以尝试任何方法来优化模型参数（确保 step() 返回展平参数）

```python
class PytorchFedAdam(FedOpt):
    """Implementation based on https://arxiv.org/abs/2003.00295."""

    def __init__(
        self,
        model: nn.Module
        params: Dict[str, npt.NDArray[np.float32]],
        learning_rate: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        t: float = 0.001,
    ) -> None:
        super().__init__(params, learning_rate, betas, t)
        self._model = model
        self._opt = torch.optim.Adam(self._model.parameters(), lr, betas=betas)

    def step(
        self, pseudo_gradient: Dict[str, npt.NDArray[np.float32]]
    ) -> Dict[str, npt.NDArray[np.float32]]:
        '''Update parameters with optimization algorithm according to pseudo gradient'''
        for name, p in self._model.named_parameters():
            p.grad = torch.from_numpy(np.array(
                            pseudo_gradient[name]
                        )).reshape(
                            p.shape
                        ).type(
                            p.grad.dtype
                        ).to(
                            p.device
                        )
        self._opt.step()
        params = dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.cpu().detach().numpy().reshape((-1))

        return params

```

### To start a fedopt server

```python
strategy = message_type.STRATEGY_FEDOPT # define server type

server = AggregateServer(args.addr, strategy, args.num)
server.run()
```
或者
```bash
python iflearner/business/homo/aggregate_server.py -n 2 --strategy FedOpt   --strategy_params {"learning_rate":1, "betas":[0.9,0.99], "t":0.1, "opt":"FedAdam"}
```

### To start a client

请参阅[如何使用](../../api/api_reference.md)

