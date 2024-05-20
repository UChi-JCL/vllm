import torch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

import vllm
from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator, loading_timer, Timer, memcpy_timer
from vllm.kvloader.simplekvloader import loader
import time
import pickle
from pathlib import Path

import vllm
from vllm.kvloader.diskkvloader import DiskKVLoader
from vllm.kvloader.cpunaiveloader import CPUKVLoader

import argparse
import yaml

vllm.core.block_manager.loader = None



def main(args):
    
    assert args.len % 2 == 0
    method2loader = {
        'regen': {},
        'cpu': CPUKVLoader('/workspace/cache/kv_transfer/', args.len // 2),
        'disk': CPUKVLoader('/workspace/cache/kv_transfer/', args.len // 2),
    }
    assert args.method in method2loader
    vllm.core.block_manager.loader = method2loader[args.method]

    # 16 tokens
    prompt = ["You are an expert in large language models, aren't you??? "] * args.len
    prompt = ' '.join(prompt)

    prompt = [prompt] * 1

    # initializing vllm
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", enable_prefix_caching=True, block_size=32, enforce_eager=True)

    # measure the end-to-end generation time.
    total_timer = Timer()

    # warm-up

    for i in range(5):
        if args.method == "cpu":
            vllm.core.block_manager.loader.read()
        total_timer.clear()
        total_timer.start()
        output = llm.generate(prompt, sampling_params)
        print(output[0].outputs[0].text)
        total_timer.pause()
        
        if i >= 2:
            with open('/workspace/vllm_demo/results/benchmark_runtime.yaml', 'a') as f:
                f.write(yaml.dump(
                    [
                        {
                            'len': args.len,
                            'ttft': total_timer.total_time,
                            'method': args.method,
                        }
                    ]
                ))
        # free cached GPU memory
        if args.method != "regen":
            del vllm.core.block_manager.loader.cache
            vllm.core.block_manager.loader.cache = None
        llm.llm_engine.scheduler.block_manager.gpu_allocator.evictor.free_table.clear()
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some inte   rs.")

    # Add the --len argument to specify the length
    parser.add_argument("--len", type=int, required=True,
                        help="Specify the length of the sequence")
    parser.add_argument("--method", type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()
    
    main(args)
