import torch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity


from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator, loading_timer, Timer, memcpy_timer
from vllm.kvloader.simplekvloader import loader

loader = {}




long_prompt = ["You are an expert school principal in JCL library oh please"] * 1333
long_prompt = ' '.join(long_prompt)


sampling_params = SamplingParams(temperature=0, max_tokens=1)
llm = LLM(model="TheBloke/CodeLlama-34B-AWQ", enable_prefix_caching=False, quantization='AWQ',enforce_eager=True,dtype="float16", block_size=128)
# llm = LLM(model="codellama/CodeLlama-34b-hf", enable_prefix_caching=False ,enforce_eager=True, dtype="float16")
# llm = LLM(model="mistralai/Mistral-7B-v0.1", enable_prefix_caching=False, enforce_eager=True)

# Create an LLM.
total_timer = Timer()
total_timer.start()
for i in range(5):
    output = llm.generate(long_prompt[i:], sampling_params)

total_timer.pause()
print('Total inference time: ', total_timer.total_time)
print(output[0].outputs[0].text)    # should be " You"

