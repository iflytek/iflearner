#!/bin/bash
for i in {1..3}; do
    echo "Starting client $i"
    python imagenet.py -a resnet18 /data1/shengxu8/projects/iflytek/data/tiny-imagenet-200-iid-$i --name iid-client0$i --epochs 60 --server "0.0.0.0:50001" --peers "0.0.0.0:50012;0.0.0.0:50013;0.0.0.0:50014" --pretrained --gpu 0 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
