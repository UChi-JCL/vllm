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
#disk="/local/"
# disk loader
# vllm.core.block_manager.loader = DiskNaiveLoader("/workspace/share/cache_layer_mistral/kvcache_regroup")
# vllm.core.block_manager.loader = DiskKVLoader(f"{disk}/share/cache_layer_mistral/kvcache")
# cachegen loader
vllm.core.block_manager.loader = DiskCachegenLoader(f"{disk}/share/cache_layer_mistral/kvcache_compressed/")
# naive load
# vllm.core.block_manager.loader = {}




# 15 tokens x 1000
long_prompt = ["You are an expert in large language models. Aren't you?? "] * 2000
long_prompt = ' '.join(long_prompt)



sampling_params = SamplingParams(temperature=0, max_tokens=1)
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", enable_prefix_caching=True, enforce_eager=True, block_size=32)

# # kv cache: disk ---> memory
# total_timer = Timer()
# total_timer.start()
# st = time.time()
# kv_path = Path('/local/kuntai/cache/kvcache')
# for file in tqdm(list(kv_path.iterdir())):
#     loader.cache[int(file.stem)] = torch.load(file)
# print('Disk -> CPU time: ', time.time() - st)


import cProfile
profiler = cProfile.Profile()

import transformers
# import line_profiler
# prof = line_profiler.LineProfiler()
# prof.add_function( vllm.transformers_utils.tokenizer.detokenize_incrementally)
# prof.add_function(transformers.PreTrainedTokenizerFast.convert_ids_to_tokens)
# prof.add_function(llm.llm_engine._decode_logprobs)
# prof.enable_by_count()

total_timer = Timer()
total_timer.start()
# implicitly load kv cache to GPU in `allocate` function, inside vllm/core/block_manager.py
# to search that part of code, grep loading_timer
profiler.runctx('llm.generate(long_prompt, sampling_params)', globals(), locals())
# output = llm.generate(long_prompt, sampling_params)
total_timer.pause()

# prof.disable_by_count()
# prof.print_stats()
# if vllm.core.block_manager.loader == {}:
#     profiler.dump_stats('profiles/recomp.prof')
# elif isinstance(vllm.core.block_manager.loader, DiskNaiveLoader):
#     profiler.dump_status('profiles/naive.prof')
# elif isinstance(vllm.core.block_manager.loader, DiskCachegenLoader):
profiler.dump_stats('profiles/cachegen.prof')
print('Total time: ', total_timer.total_time)
print('Loading time (CPU -> GPU + GPU memcpy)', loading_timer.total_time)
print('GPU memcpy time: ', memcpy_timer.total_time)
# print(output[0].outputs[0]) # should be " You"
