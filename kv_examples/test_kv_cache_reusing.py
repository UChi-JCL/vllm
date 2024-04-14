import torch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity


from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator, loading_timer, Timer, memcpy_timer
from vllm.kvloader.simplekvloader import loader

# with open('/dataheart/yuhanl-share/llmlingua_lc/0.txt', 'r') as f:
#     long_prompt = f.read()
    
# long_prompt = long_prompt + long_prompt
# breakpoint()

long_prompt = ["You are an expert school principal in JCL library oh please"] * 1333
long_prompt = ' '.join(long_prompt)

total_timer = Timer()


sampling_params = SamplingParams(temperature=0, max_tokens=1)
llm = LLM(model="lmsys/longchat-7b-16k", enable_prefix_caching=True, enforce_eager=True)

# Create an LLM.
total_timer.start()
print('Start timing..................')
# with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
output = llm.generate(long_prompt, sampling_params)
# prof.export_chrome_trace("trace.json")
total_timer.pause()
print('Loading time: ', loading_timer.total_time)
print('Total time: ', total_timer.total_time)
print('Memcpy time: ', memcpy_timer.total_time)
loading_timer.clear()
total_timer.clear()
memcpy_timer.clear()

breakpoint()




# # get blocks
block_manager = llm.llm_engine.scheduler.block_manager.gpu_allocator
cache_engine = CacheEngineManager.GetCacheEngine()
gpu_cache = cache_engine.gpu_cache

hashed_blocks = block_manager.evictor.free_table
cachegen = {}

for hash in tqdm(list(hashed_blocks.keys())):
    block = hashed_blocks[hash]
    
    caches = []
    
    for gpu_cache_layer in cache_engine.gpu_cache:
        k_cache = gpu_cache_layer[0][block.block_number]
        v_cache = gpu_cache_layer[1][block.block_number]
        
        num_heads, head_size, n_tokens = v_cache.shape
        
        k = k_cache.permute(2,0,1,3).reshape(n_tokens, num_heads, head_size)
        v = v_cache.permute(2,0,1)
        # corresponding = gpu_cache_layer[block.block_number]
        # caches.append((k.to('cpu').pin_memory(), v.to('cpu').pin_memory()))
        caches.append((k.to('cpu'), v.to('cpu')))

    loader[hash] = caches

    

# clear hash table
for hash in list(hashed_blocks):
    del hashed_blocks[hash]
    
for gpu_cache_layer in cache_engine.gpu_cache:
    gpu_cache_layer[0][:] = 0.
    gpu_cache_layer[1][:] = 0.
    
        
# now this should be faster.
total_timer.start()
output = llm.generate(long_prompt, sampling_params)
total_timer.pause()
print('Loading time: ', loading_timer.total_time)
print('Total time: ', total_timer.total_time)
print('Memcpy time: ', memcpy_timer.total_time)
loading_timer.clear()
total_timer.clear()
memcpy_timer.clear()

# breakpoint()