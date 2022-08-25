## qFedAvg
According to the article [***FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING***](https://openreview.net/pdf?id=ByexElSYDr), we implement qfedavg aggregation algorithm.

The complete source code 
reference for this example [qFedAvg](https://git.iflytek.com/TURING/iflearner/-/tree/master/examples/homo/qFedAvg).
### To start a qfedav server

```python
strategy = message_type.STRATEGY_qFEDAVG # define server type

server = AggregateServer(args.addr, strategy, args.num, q=.2, learning_rate = 1)
server.run()
```
or
```shell
python iflearner/business/homo/aggregate_server.py -n 2 --strategy qFedAvg   --strategy_params {"q":.2, "learning_rate":1}
```

### To start a client

See [how to use](../../api/api_reference.md)

It should be noted that before the client starts fitting, you need to obtain the loss value `loss` of the current model on the training data. Then, you should override the `Trainer.get` method on the client side and add the loss keyword to the uploaded parameters.

```python
def evaluate_traindata(self):
    batch_time = AverageMeter("Time", ":6.3f", Summary.AVERAGE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(self._train_loader), [batch_time, losses, top1, top5], prefix="Test on training data: "
    )

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(self._train_loader):
            if self._args.gpu is not None:
                images = images.cuda(self._args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(self._args.gpu, non_blocking=True)

            # compute output
            output = self._model(images)
            loss = self._criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self._args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
    self._fs = losses.avg

def get(self, param_type=''):
    parameters = dict()
    parameters['loss'] = np.array([self._fs])
    for name, p in self._model.named_parameters():
        if p.requires_grad:
            parameters[name.replace('module.', '')
                        ] = p.cpu().detach().numpy()

    return parameters
```

