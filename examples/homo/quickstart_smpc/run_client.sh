#!/bin/bash


python quickstart_pytorch.py --name "client1" --epochs 10 --server "0.0.0.0:50001" --peers "0.0.0.0:50012;0.0.0.0:50013" &
python quickstart_pytorch.py --name "client2" --epochs 10 --server "0.0.0.0:50001" --peers "0.0.0.0:50013;0.0.0.0:50012" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
