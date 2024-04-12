
import torch
from tqdm import tqdm

keys = torch.load('/local/share/hash_sequence.pt')
keys = torch.tensor(keys)

CHUNK_SIZE=3200
keymap = {}

for idx, regrouped_keys in tqdm(enumerate(torch.split(keys, CHUNK_SIZE//32))):
    
    # read all tensors
    tensors = [
        torch.load(f'/local/share/cache/kvcache/{key}.pt')
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
        
    torch.save(tensors, f'/local/share/cache/kvcache_regroup/{idx}.pt')
    
    
    

torch.save(keymap, '/local/share/cache/kvcache_regroup/keymap.pt')
torch.save(keymap, '/local/share/cache_layer/kvcache_compressed/keymap.pt')