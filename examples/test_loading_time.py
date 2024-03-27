
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# data = torch.load('kv_cache.pt')
# ks, vs = [], []

# for layers in data.cache.values():
#     for layer in layers:
#         k, v = layer
#         ks.append(k)
#         vs.append(v)
        
# ks, vs = torch.cat(ks), torch.cat(vs)

# uncompressed: 6.3 GB
# transfer time: 2.6 sec
# bandwidth: 2.5 GBps --> 20 Gbps

# torch.save(ks, '/local/kuntai/cache/ks.pt')
# torch.save(vs, '/local/kuntai/cache/vs.pt')
# ks = torch.load('/local/kuntai/cache/ks.pt').pin_memory()
# vs = torch.load('/local/kuntai/cache/vs.pt').pin_memory()
ks = torch.load('/local/kuntai/cache/ks.pt').pin_memory()
vs = torch.load('/local/kuntai/cache/vs.pt').pin_memory()
# ks = torch.load('/local/kuntai/cache/ks.pt')
# vs = torch.load('/local/kuntai/cache/vs.pt')
# ks = torch.load('/dataheart/kuntai_recovery/code/vllm/ks.pt').pin_memory()
# vs = torch.load('/dataheart/kuntai_recovery/code/vllm/vs.pt').pin_memory()


print(ks.device)
print(vs.device)
print(ks.shape)
print(vs.shape)
print(ks.abs().mean())
print(vs.abs().mean())


import time


# with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
for i in range(10):
    st = time.time()
    tmp = ks.cuda(), vs.cuda()
    print(time.time() - st)
    
# torch.cuda.synchronize()

# prof.export_chrome_trace("trace.json")

# 2.16s w/ cuda sync
# 2.21
# heart: 3.93 s, k & v