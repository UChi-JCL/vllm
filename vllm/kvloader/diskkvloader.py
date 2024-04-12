
from vllm.kvloader.kvloader_base import KVLoaderBase, KVCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import time
from pathlib import Path


class DiskKVLoader(KVLoaderBase):
    def __init__(self, root: str):
        self.root = Path(root)

    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        torch.save(kvcache, self.root / f"{hash}.pt")
    # @profile
    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:
        file = self.root / f"{hash}.pt"
        st = time.monotonic()
        if file.exists():
            out = torch.load(file)
            return out
        else:
            return None
    
    def __contains__(self, hash: int) -> bool:
        file = self.root / f"{hash}.pt"
        return file.exists()

