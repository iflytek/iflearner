#!/bin/bash

for i in $(seq 2); do
    echo "Starting client $i"
    python quickstart_keras.py --name "client$i" --epochs 3 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait