from src.encoder.encoder import CacheGenEncoder
import torch
import pickle
import torchac
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
p = argparse.ArgumentParser()
p.add_argument("--path_to_encoded_kv", type=str)
p.add_argument("--quantization_config", type=str)
p.add_argument("--model_config", type=str)
p.add_argument("--chunk_size", type=int)
p.add_argument("--model_id", type=str)
p.add_argument("--input_text", type=str)
p.add_argument("--path_to_original_kv", type=str)
args = p.parse_args()



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
def transform_tuple_to_tensors(kv):
    """ Given a tuple of key-value tensors, transform them into
    a single tensor of shape (num_layers, tokens, num_heads * head_dim)
    """
    kv = kv.unsqueeze(dim=2)
    head_num = kv[0][0].shape[1]
    head_dim = kv[0][0].shape[3]
    tokens_num = kv[0][0].shape[2]
    k_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    v_tensor = torch.zeros((len(kv), tokens_num, head_num * head_dim))
    for i in range(len(kv)):
        k_tensor[i] = kv[i][0].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
        v_tensor[i] = kv[i][1].permute(0, 2, 1, 3).reshape(tokens_num, head_num * head_dim)
    return k_tensor, v_tensor
def concat_dict(dict1, start, end):
    """ Concat the dict of CDF into a single tensor, start is
    the start_layher, end is the end_layer
    """
    concat_tensor = None
    for i in range(start, end):
        if concat_tensor is None:
            concat_tensor = dict1[i].unsqueeze(0)
        else:
            concat_tensor = torch.cat((concat_tensor, \
                dict1[i].unsqueeze(0)), dim=0)
    return concat_tensor
def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)
def encode_function(path_to_original_kv, quantization_config, CHUNK_SIZE, output_path):
    """
    Given the path to the original key value cache, encode the KV cache
    Fields:
    - path_to_original_kv: the path to the original key value cache
    - quantization_config: the path to the quantization config
    - CHUNK_SIZE: the chunk size to encode, NEEDS to be multiples of 20!!!
    - output_path: the path to the output file
    """
    output_dict = {}
    config = json.load(open(quantization_config, "r"))
    test_kv = torch.load(path_to_original_kv)
    # test_kv = pickle.load(open(path_to_original_kv, "rb"))
    X = test_kv
    fp_k = X[:, 0].reshape((X.shape[0], X.shape[2], X.shape[-2] * X.shape[-1]))
    fp_v = X[:, 1].reshape((X.shape[0], X.shape[2], X.shape[-2] * X.shape[-1]))
    l = fp_k.shape[0]
    encoder = CacheGenEncoder(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize(config=config)
    cdf_k = encoder.compute_cdf(is_key=True, config=config)
    encode_input_key = concat_dict(encoder.quantized_key, 0, config["key_first_layers"])
    encode_input_key = torch.cat((encode_input_key, \
        concat_dict(encoder.quantized_key, config["key_first_layers"], config["key_second_layers"]) ), dim=0)
    encode_input_key = torch.cat((encode_input_key, \
        concat_dict(encoder.quantized_key, config["key_second_layers"], l) ), dim=0)
    cdf_v = encoder.compute_cdf(is_key=False, config=config)
    encode_input_value = concat_dict(encoder.quantized_value, 0, config["value_first_layers"])
    encode_input_value = torch.cat((encode_input_value, \
        concat_dict(encoder.quantized_value, config["value_first_layers"], l) ), dim=0)
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0)
    # # cdf = cdf.unsqueeze(2).repeat(1, 1, 4096, 1)
    # print(encode_input.shape, cdf.shape)
    
    st = time.monotonic()
    bitstreams = b""
    maxsize = 1024 * 160 * CHUNK_SIZE
    encode_function.BUFFER = np.zeros(maxsize, dtype=np.uint8)
    buffer = encode_function.BUFFER
    current_index = 0
    start_indices = []
    for l in range(cdf.shape[0]):
        print("Done with layer", l)
        for i in range(CHUNK_SIZE):
            bits = torchac.encode_float_cdf(cdf[l:l+1], \
                encode_input[l:l+1, i].to(torch.int16) )
            # start_indices += [len(bitstreams)]
            # bitstreams += bits
            length = len(bits)
            start_indices += [current_index]
            buffer[current_index:current_index + length] = np.frombuffer(bits, dtype=np.uint8)
            current_index += length
    print("Time to encode", time.monotonic() - st)
    # output_dict["bitstreams"] = bitstreams
    output_dict[f"bitstreams"] = torch.ByteTensor(list(buffer[:current_index].tobytes()))
    output_dict[f"start_indices"] =  torch.tensor(start_indices).int()
    output_dict["cdf"] = _renorm_cast_cdf_(cdf.float(), 16)
    output_dict["max_tensors_key"] = concat_max(encoder.max_tensors_key)
    output_dict["max_tensors_value"] = concat_max(encoder.max_tensors_value)
    pickle.dump(output_dict, open(output_path, "wb"))
encode_function.BUFFER = None
if __name__ == "__main__":
    for i in tqdm(list(range(5))):
        encode_function(
            f"/local/share/cache_layer/kvcache_regroup/{i}.pt",
            "./config/quantization_7b.json",
            3200,
            f"/local/share/cache_layer/kvcache_compressed/{i}.pkl",
        )
    # encode_function(args.path_to_original_kv, args.quantization_config, args.chunk_size, args.path_to_encoded_kv)
