## FedNova
According to the article [***Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization***
](https://arxiv.org/abs/2007.07481), we implement FedNova aggregation algorithm.

The complete source code 
reference for this example [fednova](https://git.iflytek.com/TURING/iflearner/-/tree/master/examples/homo/fednova).
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

FedNova involves the number of samples on the client side (`sample_num`) and the number of times each round of training optimizations (`batch_num`), so the `Trainer.config` method needs to be override to return these two values

```python
def config(self) -> dict():
        
    return {
        "batch_num": len(self._train_loader),
        "sample_num": len(self._train_loader) * self._train_loader.batch_size,
    }
```

FedNova also needs to override the `Trainer.get` method to return the difference between the client current model and the previous round model.

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