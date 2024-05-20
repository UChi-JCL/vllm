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
from vllm.kvloader.diskcachegenloader import DiskCachegenLoader
from vllm.kvloader.disknaiveloader import DiskNaiveLoader

disk="/workspace/"
# disk loader
# vllm.core.block_manager.loader = DiskNaiveLoader("/workspace/share/cache_layer_longchat/kvcache_regroup")
# cachegen loader
vllm.core.block_manager.loader = DiskCachegenLoader(f"/workspace/share/cache_layer_longchat/kvcache_compressed/")
# naive load
# vllm.core.block_manager.loader = {}




# 15 tokens x 1000
long_prompt = ""
for i in range(4):
    source = f"./CacheGen/7k_prompts/{i}.txt"
    if i >=2:
        source = f"./CacheGen/9k_prompts/{i}.txt"
    with open(source, "r") as f:
        long_prompt = long_prompt + f.read()
    long_prompt = long_prompt + '\n'
    
# long_prompt = long_prompt[:1000]
    
long_prompt = long_prompt + "Tell me what was the first topic we discussed. Use at most 5 words."



sampling_params = SamplingParams(temperature=0, max_tokens=100)
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", enable_prefix_caching=True, block_size=32, enforce_eager=True)
print("vLLM profiling finished\n\n\n")

import cProfile
profiler = cProfile.Profile()
profiler = None

import transformers

total_timer = Timer()
total_timer.start()

if profiler is not None:
    profiler.runctx('llm.generate(long_prompt, sampling_params)', globals(), locals())
else:
    output = llm.generate(long_prompt, sampling_params)
total_timer.pause()
print('Total time: ', total_timer.total_time)

if profiler is not None:
    if vllm.core.block_manager.loader == {}:
        profiler.dump_stats('profiles/recomp.prof')
    elif isinstance(vllm.core.block_manager.loader, DiskNaiveLoader):
        profiler.dump_stats('profiles/naive.prof')
    elif isinstance(vllm.core.block_manager.loader, DiskCachegenLoader):
        profiler.dump_stats('profiles/cachegen.prof')

print(output[0].outputs[0].text) # should be "The role of art in society.</s>"
