
from vllm.kvloader.kvloader_base import KVLoaderBase, KVCache
from typing import Any, Dict, List, Optional, Tuple


class SimpleKVLoader(KVLoaderBase):
    def __init__(self):
        self.cache: Dict[int, KVCache] = {}

    def __setitem__(self, hash: int, kvcache: KVCache):
        self.cache[hash] = kvcache

    def __getitem__(self, hash: int) -> Optional[KVCache]:
        return self.cache.get(hash)
    
    def __contains__(self, hash: int) -> bool:
        return hash in self.cache

loader = SimpleKVLoader()
