
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

def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf
def quant(xq, max1, dim=-1, quant_type="vector"):
    
    C = int(os.environ["BINS"]) // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x


def decode_function(path_to_encoded_kv, quantization_config, model_config, CHUNK_SIZE):
    """
    Given the path to the encoded key value cache, decode the KV cache
    Fields:
    - path_to_encoded_kv: the path to the encoded key value cache
    - quantization_config: the path to the quantization config
    - model_config: the path to the model config
    - CHUNK_SIZE: the chunk size to decode, NEEDS to be multiples of 20!!! 
    Outputs:
    - key: the decoded key tensor in the shape of (layers, num_heads, tokens, heads_dim)
    """
    config = quantization_config
    encoded_file = pickle.load(open(path_to_encoded_kv, "rb"))
    cdf = encoded_file["cdf"]
    cdf = _renorm_cast_cdf_(cdf.float(), 16)
    output = torch.zeros( (CHUNK_SIZE, cdf.shape[0] * model_config['hidden_dim'] )).cuda().to(torch.int)
    bits = encoded_file["bitstreams"]
    concated_string = bits
    start_indices= encoded_file["start_indices"]
    max_tensors_k = encoded_file["max_tensors_key"]
    max_tensors_v = encoded_file["max_tensors_value"]
    out = torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string.tobytes(), \
        start_indices, CHUNK_SIZE, 20, CHUNK_SIZE//20)
    # out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  6000, 60, 100)
    out = output.reshape((CHUNK_SIZE, 2, max_tensors_k.shape[0], \
        model_config["hidden_dim"])).permute(1, 2, 0, 3)
    key = out[0].half()
    value = out[1].half()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            os.environ['BINS'] = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            os.environ['BINS'] = config["key_second_bins"]
        else:
            os.environ['BINS'] = config["key_third_bins"]
        key[l] = quant(key[l] - int(os.environ['BINS']) // 2 + 1, \
            max_tensors_k[l, :CHUNK_SIZE].cuda()).clone()
    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            os.environ['BINS'] = config["value_first_bins"]
        else:
            os.environ['BINS'] = config["value_second_bins"]
        value[l] = quant(value[l] - (int(os.environ['BINS']) // 2- 1), \
            max_tensors_v[l, :CHUNK_SIZE].clone().cuda()).clone()
    key = key.reshape(
        key.shape[0],
        1,
        key.shape[1],
        model_config["num_heads"],
        model_config["heads_dim"])
    value = value.reshape(
        value.shape[0],
        1,
        value.shape[1],
        model_config["num_heads"],
        model_config["heads_dim"])
    return torch.cat([key, value], dim=1)

class DiskCachegenLoader(KVLoaderBase):
    def __init__(self, root: str):
        self.root = Path(root)
        self.config = torch.load(self.root / "keymap.pt")
        self.cached_tensor = {}

        self.model_config = json.load(open("/dataheart/kuntai_recovery/code/CacheGen/config/codellama_34b_awq.json", "r"))
        self.quant_config = json.load(open("/dataheart/kuntai_recovery/code/CacheGen/config/quantization_7b.json", "r"))

    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        raise NotImplementedError

    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:
        idx = self.config[hash]["idx"]
        offset = self.config[hash]["offset"]
        print("idx", idx, "offset", offset, "hash", hash)
        if idx not in self.cached_tensor:
            print("reallocating")
            del self.cached_tensor
            self.cached_tensor = {}
            # self.cached_tensor[idx] = torch.load(f"/local/kuntai/cache/kvcache_regroup/{idx}.pt")
            self.cached_tensor[idx] = decode_function(
                f"{self.root}/{idx}.pkl",
                self.quant_config,
                self.model_config,
                16000
            )
        return self.cached_tensor[idx][:, :, offset:offset+32, :, :]
    
    def __contains__(self, hash: int) -> bool:
        return hash in self.config

