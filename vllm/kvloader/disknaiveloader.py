
from vllm.kvloader.kvloader_base import KVLoaderBase, KVCache
from typing import Any, Dict, List, Optional, Tuple
import torch
from pathlib import Path
import torch
import torchac_cuda

import torch 
import pickle
import os
import torchac_cuda
import time
import json
import argparse


class DiskNaiveLoader(KVLoaderBase):
    def __init__(self, root: str):
        self.root = Path(root)
        self.config = torch.load(self.root / "keymap.pt")
        self.cached_tensor = {}

        self.output = torch.zeros( (3200, 96 * 1024 )).cuda().to(torch.int)
        self.dataloader = None
    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        raise NotImplementedError

    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:

        keymap = self.config[hash]
        idx, offset = keymap["idx"], keymap["offset"]
        # print("idx", idx, "offset", offset, "hash", hash)
        
        if idx not in self.cached_tensor:
            st = time.monotonic()
            del self.cached_tensor
            self.cached_tensor = {}
            self.cached_tensor[idx] = torch.load(f"{self.root}/{idx}.pt").cuda()
            print(f"Loaded {self.root}/{idx}.pt")
            print("Finished loading one file: ", time.monotonic() - st)
        #
        return self.cached_tensor[idx][:, :, offset:offset+32, :, :]
    
    def __contains__(self, hash: int) -> bool:
        return hash in self.config

