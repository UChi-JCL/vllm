
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
# 

def decode_function(encoded_file, quantization_config, model_config, CHUNK_SIZE, output):
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
    start_time = time.monotonic()
    # encoded_file = pickle.load(open(encoded_file, "rb"))
    # print("loading delay: ", time.monotonic() - st)
    cdf = encoded_file["cdf"]
    # cdf = _renorm_cast_cdf_(cdf.float(), 16)
    bits = encoded_file["bitstreams"]
    concated_string = bits
    start_indices= encoded_file["start_indices"]
    max_tensors_k = encoded_file["max_tensors_key"]
    max_tensors_v = encoded_file["max_tensors_value"]
    # concated_string = 
    
    nlayers = cdf.shape[0]
    scale = 4
    # output = torch.zeros( ( CHUNK_SIZE * 96 * 1024 )).cuda().to(torch.int)
    # out = torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string, \
    #     start_indices, CHUNK_SIZE, 32, CHUNK_SIZE//32)st = time.monotonic()
    kernel_start = time.monotonic()
    start_indices = torch.tensor(start_indices).int().cuda()
    out = torchac_cuda.decode_fast(output, cdf.unsqueeze(0), concated_string, \
        start_indices, CHUNK_SIZE, nlayers * scale, 800, scale)
    torch.cuda.synchronize()
    print("kernel computation time: ", time.monotonic() - kernel_start)
    # out = torchac_cuda.decode(output, cdf.unsqueeze(0), bits,  6000, 60, 100)
    out = output.reshape((2, max_tensors_k.shape[0], CHUNK_SIZE, \
        model_config["hidden_dim"]))
    key = out[0].half()
    value = out[1].half()
    max_tensors_k = max_tensors_k.pin_memory()
    max_tensors_k = max_tensors_k.cuda()
    max_tensors_v = max_tensors_v.pin_memory()
    max_tensors_v = max_tensors_v.cuda()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            os.environ['BINS'] = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            os.environ['BINS'] = config["key_second_bins"]
        else:
            os.environ['BINS'] = config["key_third_bins"]
        key[l] = quant(key[l] - int(os.environ['BINS']) // 2 + 1, max_tensors_k[l, :CHUNK_SIZE])
    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            os.environ['BINS'] = config["value_first_bins"]
        else:
            os.environ['BINS'] = config["value_second_bins"]
        value[l] = quant(value[l] - (int(os.environ['BINS']) // 2- 1), max_tensors_v[l, :CHUNK_SIZE] )
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
    out = torch.cat([key, value], dim=1)
    torch.cuda.synchronize()
    print("per iteration total time: ", time.monotonic() - start_time)
    return out



class CachegenDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_samples):
        """
        Args:
            data_dir (string): Directory with the .pt files.
            num_samples (int): Total number of samples/files.
        """
        self.data_dir = data_dir
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Construct the file path from the directory and idx
        file_path = f"{self.data_dir}/{idx}.pkl"
        
        # Load and return the tensor stored in the file
        data = pickle.load(open(file_path, "rb"))
        return data





class DiskCachegenLoader(KVLoaderBase):
    def __init__(self, root: str):
        self.root = Path(root)
        self.config = torch.load(self.root / "keymap.pt")
        self.cached_tensor = {}

        self.model_config = json.load(open("/local/share/config/codellama_34b_awq.json", "r"))
        self.quant_config = json.load(open("/local/share/config/quantization_7b.json", "r"))
        self.output = torch.zeros( (3200, 96 * 1024 )).cuda().to(torch.int)
        self.dataloader = None
    def __setitem__(self, hash: int, kvcache: torch.Tensor):
        raise NotImplementedError
    
    def __getitem__(self, hash: int) -> Optional[torch.Tensor]:
        idx = self.config[hash]["idx"]
        offset = self.config[hash]["offset"]
        # print("idx", idx, "offset", offset, "hash", hash)

        if idx not in self.cached_tensor:

            st = time.monotonic()

            if self.dataloader is None:
                self.dataloader = iter(torch.utils.data.DataLoader(CachegenDataset(data_dir=self.root, num_samples=5), batch_size=1, shuffle=False, num_workers=1, ))
            print("reallocating")
            del self.cached_tensor
            self.cached_tensor = {}
            print("IDX: ", idx)
            encoded_data = next(self.dataloader)
            for key in encoded_data:
                # remove the batch dimension.
                encoded_data[key] = encoded_data[key].squeeze(0)
            self.cached_tensor[idx] = decode_function(
                # f"{self.root}/{idx}.pkl",
                encoded_data,
                self.quant_config,
                self.model_config,
                3200, 
                self.output
            )
            print('End-to-end time: ', time.monotonic() - st)
        return self.cached_tensor[idx][:, :, offset:offset+32, :, :]
    
    def __contains__(self, hash: int) -> bool:
        return hash in self.config

