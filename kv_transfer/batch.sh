#!/bin/bash

set -xe

rm -f /workspace/vllm_demo/results/benchmark_runtime.yaml

# Array containing the methods
methods=("regen" "cpu" "disk")
# methods=("cpu")


# Loop over the range of lengths
for len in 2 4 8 16 32 64 128 256 512 1024 2000;
do

    # Loop over each method and execute the command
    for method in "${methods[@]}"
    do
        python kv_cache_benchmark.py --method $method --len $len
    done

    for method in "${methods[@]}"
    do
        python kv_cache_benchmark.py --method $method --len $len
    done

done
