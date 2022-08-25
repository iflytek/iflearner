## FedNova
根据论文 [***Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization***
](https://arxiv.org/abs/2007.07481), 我们实现了FedNova聚合算法

完整的该示例的源代码参考[FedNova](https://git.iflytek.com/TURING/iflearner/-/tree/master/examples/homo/fednova)。

### To start a FedNova server

```python
strategy = message_type.STRATEGY_FEDNOVA # define server type

server = AggregateServer(args.addr, strategy, args.num)
server.run()
```
or
```shell
python iflearner/business/homo/aggregate_server.py -n 2 --strategy FedNova 
```

### To start a client

See [how to use](../../api/api_reference.md)

FedNova涉及客户端的样本数（`sample_num`）和每轮训练优化的次数（`batch_num`），因此需要覆盖`Trainer.config`方法以返回这两个值


```python
def config(self) -> dict():
        
    return {
        "batch_num": len(self._train_loader),
        "sample_num": len(self._train_loader) * self._train_loader.batch_size,
    }
```

FedNova 还需要重写 `Trainer.get`方法，以返回客户端当前模型和上一轮模型之间的差异。



```python
def get(self, param_type=""):
        parameters = dict()
        for name, p in self._model.named_parameters():
            if p.requires_grad:
                parameters[name.replace("module.", "")] = (
                    p.cpu().detach().numpy()
                    - self._old_weights[name].cpu().detach().numpy()
                )

        return parameters

```