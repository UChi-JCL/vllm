
import torch
from tqdm import tqdm
from collections import defaultdict 

keys = torch.load('hash_sequence.pt')
keys = torch.tensor(keys)

CHUNK_SIZE=3200
keymap = {}

for idx, regrouped_keys in tqdm(enumerate(torch.split(keys, CHUNK_SIZE//32))):
    # read all tensors
    tensors = [
        torch.load(f'/workspace/share/cache_layer_mistral/kvcache/{key}.pt')
        for key in regrouped_keys
    ]
    
    tensors = torch.cat(tensors, dim=2)

    if tensors.shape[2] != CHUNK_SIZE:
        zeros = torch.zeros(
            tensors.shape[0], 
            tensors.shape[1], 
            CHUNK_SIZE - tensors.shape[2],
            tensors.shape[3],
            tensors.shape[4],
            dtype=tensors.dtype)
        tensors = torch.cat([tensors, zeros], dim=2)
    
    offset = 0
    for key in regrouped_keys:
        keymap[key.item()] = {
            "idx": idx,
            "offset": offset,
        }
        offset += 32
        
    torch.save(tensors, f"/workspace/share/cache_layer_mistral/kvcache_regroup/{idx}.pt")
    
    

torch.save(keymap, '/workspace/share/cache_layer_mistral/kvcache_regroup/keymap.pt')
torch.save(keymap, '/workspace/share/cache_layer_mistral/kvcache_compressed/keymap.pt')
