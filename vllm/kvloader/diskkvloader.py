
from vllm.kvloader.kvloader_base import KVLoaderBase, KVCache
from typing import Any, Dict, List, Optional, Tuple
import torch
from pathlib import Path


class DiskKVLoader(KVLoaderBase):
    def __init__(self, root: str):
        self.root = Path(root)

    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        torch.save(kvcache, self.root / f"{hash}.pt")

    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:
        file = self.root / f"{hash}.pt"
        if file.exists():
            return torch.load(file)
        else:
            return None
    
    def __contains__(self, hash: int) -> bool:
        file = self.root / f"{hash}.pt"
        return file.exists()

