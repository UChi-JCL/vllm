
import torch
from pathlib import Path
import time
import concurrent.futures
from tqdm import tqdm
import numpy as np

# without pipelining
kv_path = Path('/local/kuntai/cache/kvcache')

cachegen = {}


st = time.time()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
futures = []
for file in tqdm(list(kv_path.iterdir())):
    futures.append(executor.submit(lambda x: torch.load(x).cuda(), file))
executor.shutdown(wait=True)
print('Total time: ', time.time() - st)


st = time.time()

for file in tqdm(list(kv_path.iterdir())):
    hash_value = int(file.stem)
    cachegen[hash_value] = torch.load(file)
    
print('Disk IO time: ', time.time() - st)
st = time.time()
    
    
for hash_value in tqdm(list(cachegen.keys())):
    cachegen[hash_value].cuda()
    
print(np.mean([t.mean() for t in cachegen.values()]))

print('CPU -> GPU time: ', time.time() - st)



        
    