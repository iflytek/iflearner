# qFedavg example

In the example, we will introduce how to train imagenet with Ifleaner qFedavg and PyTorch. More details about qFedavg can be seen [here](../../../doc/source/qfedavg.md)

### Server

```cmd
python server.py -n 2 --server "0.0.0.0:50001"
```

### Client

```cmd
python imagenet.py --data your_data_path1 --name client01 --epochs 60 --server "0.0.0.0:50001"
python imagenet.py --data your_data_path2 --name client02 --epochs 60 --server "0.0.0.0:50001"
```
