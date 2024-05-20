
from vllm.kvloader.kvloader_base import KVLoaderBase, KVCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import time
from pathlib import Path
import pickle


class CPUKVLoader(KVLoaderBase):
    def __init__(self, root: str, num: int):
        self.root = Path(root)
        self.num = num
        self.cache = None
        
        self.seq = torch.load('/workspace/vllm_demo/kv_transfer/hash_sequence.pt')
        self.hash_to_idx = {}
        
        for idx in range(num):
            self.hash_to_idx[self.seq[idx]] = idx
        
    def read(self):
        del self.cache
        self.cache = []
        
        # self.cache = torch.load('/workspace/cache/kv_transfer_full.pt')
        filename = f'/workspace/cache/kv_transfer_whole/{self.num}.pickle'
        
        if not Path(filename).exists():
            
            print(f"Reconstructing {filename}")
                
            for idx in range(self.num):
                hash = self.seq[idx]
                with open(self.root / f"{hash}.pickle", 'rb') as f:
                    out = pickle.load(f)
                self.cache.append(out.unsqueeze(0))
            self.cache = torch.cat(self.cache, dim=0)
            
            with open(filename, 'wb') as f:
                pickle.dump(self.cache, f)
                
        with open(filename, 'rb') as f:
            self.cache = pickle.load(f)

    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        torch.save(kvcache, self.root / f"{hash}.pickle")
        
        
    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:
        if self.cache is None:
            self.read()
        if self.cache.is_cpu:
            self.cache = self.cache.cuda()
            
        return self.cache[self.hash_to_idx[hash]]
    
    def __contains__(self, hash: int) -> bool:
        return hash in self.hash_to_idx

