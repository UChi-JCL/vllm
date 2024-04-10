import torch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity


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

# # disk loader
# vllm.core.block_manager.loader = DiskKVLoader("/local/kuntai/cache/kvcache")
# cachegen loader
vllm.core.block_manager.loader = DiskCachegenLoader("/local/kuntai/cache/kvcache_compressed")



# 15 tokens x 1000
long_prompt = ["You are an expert in large language models, aren't you? "] * 1000
long_prompt = ' '.join(long_prompt)




sampling_params = SamplingParams(temperature=0, max_tokens=1)
llm = LLM(model="TheBloke/CodeLlama-34B-AWQ", enable_prefix_caching=True, enforce_eager=True, quantization='AWQ',dtype="float16", block_size=32)


# # kv cache: disk ---> memory
# total_timer = Timer()
# total_timer.start()
# st = time.time()
# kv_path = Path('/local/kuntai/cache/kvcache')
# for file in tqdm(list(kv_path.iterdir())):
#     loader.cache[int(file.stem)] = torch.load(file)
# print('Disk -> CPU time: ', time.time() - st)



total_timer = Timer()
total_timer.start()
# implicitly load kv cache to GPU in `allocate` function, inside vllm/core/block_manager.py
# to search that part of code, grep loading_timer
output = llm.generate(long_prompt, sampling_params)
total_timer.pause()
print('Total time: ', total_timer.total_time)
print('Loading time (CPU -> GPU + GPU memcpy)', loading_timer.total_time)
print('GPU memcpy time: ', memcpy_timer.total_time)
print(output[0].outputs[0]) # should be " You"
