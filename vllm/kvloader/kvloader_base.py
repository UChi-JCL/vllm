
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch

KVCache = Tuple[torch.Tensor, torch.Tensor]

class KVLoaderBase(ABC):

    @abstractmethod
    def __setitem__(hash: int, kvcache: KVCache):
        pass

    @abstractmethod
    def __getitem__(has: int) -> Optional[KVCache]:
        pass

    @abstractmethod
    def __contains__(self, hash: int) -> bool:
        pass

