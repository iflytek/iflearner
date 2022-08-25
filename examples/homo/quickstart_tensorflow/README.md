# iFlearner Example using PyTorch

This introductory example will help you understand how to run a federated task under tensorflow.

## Preconditions
First, you need to install iflearner, and then install the corresponding dependencies
```shell
pip install iflearner
```

install corresponding dependencies.
```shell
pip3 install -r requirements.txt
```

## Project Setup

First you are ready to start the iflearner server as well as the clients. You can simply start the server in a terminal as follows:.  
```shell
python3 server.py -n 2
```
> -n: The parameter n represents accept n clients to connect and you can modify it

Now you are ready to start the iflearner clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 quickstart_tensorflow.py --name "client1" --epochs 10
```

Start client 2 in the second terminal:

```shell
python3 quickstart_tensorflow.py --name "client2" --epochs 10
```

Of course, you can also quickly modify the script `run_server.sh` and `run_client.sh`, and then use the following command to quickly start an experience demo.

start server in the first terminal.
```shell
bash run_server.sh
```

start multiple clients in the second terminal.
```shell
bash run_client.sh
```