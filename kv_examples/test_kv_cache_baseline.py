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

files = list(kv_path.iterdir())
files = files + files + files

for file in files:
    hash_value = int(file.stem)
    torch.load(file).cuda(non_blocking=True)
    
print('CPU -> GPU time: ', time.time() - st)