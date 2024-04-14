
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import pandas as pd
import torch
from src.cachegen_engine import CacheGenController
import json
from torch.profiler import profile, record_function, ProfilerActivity
from fastchat.model import load_model
p = argparse.ArgumentParser()
def torch_quant(qA):
    # shape (8, 2048)
    MAX = int(os.environ["BINS"]) // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float32)
    
    return x, max1
def compress(kv, config):
    kv = list(kv)
    chunk_size = 1500
    for i in range(len(kv)):
        key = kv[i][0].permute(0, 2, 1, 3).reshape((1, kv[i][0].shape[2], -1))
        value = kv[i][1].permute(0, 2, 1, 3).reshape((1, kv[i][1].shape[2], -1))
        key_orig = key.clone()
        for k in range(0, key.shape[1], chunk_size):
            key_chunk = key[:, k:k+chunk_size]
            value_chunk = value[:, k:k+chunk_size]
            if list(config[config["chunk_id"] == k//1500]["quality"])[0] == 1:
                if i < 20: 
                    os.environ['BINS'] = "32"
                elif i < 40:
                    os.environ['BINS'] = "16"
                else:
                    os.environ['BINS'] = "8"
            elif list(config[config["chunk_id"] == k//1500]["quality"])[0]  == 2:
                if i < 20: 
                    os.environ['BINS'] = "32"
                else:
                    os.environ['BINS'] = "16"
            else:
                if i < 20: 
                    os.environ['BINS'] = "64"
                elif i < 40:
                    os.environ['BINS'] = "32"
                else:
                    os.environ['BINS'] = "32"
                
            key_chunk, _ = torch_quant(key_chunk)
            if list(config[config["chunk_id"] == k//1500]["quality"])[0] == 1:
                if i < 20: 
                    os.environ['BINS'] = "16"
                else:
                    os.environ['BINS'] = "8"
            elif list(config[config["chunk_id"] == k//1500]["quality"])[0]  == 2:
                if i < 10: 
                    os.environ['BINS'] = "32"
                else:
                    os.environ['BINS'] = "16"
            else:
                if i < 10:
                    os.environ['BINS'] = "64"
                else:
                    os.environ['BINS'] = "16"
            value_chunk, _ = torch_quant(value_chunk)
            key[:, k:k+chunk_size] = key_chunk
            value[:, k:k+chunk_size] = value_chunk
        loss = torch.nn.functional.mse_loss(key, key_orig)
        print(loss)
        key = key.reshape((1, key.shape[1], 32, 128)).permute(0, 2, 1, 3)
        value = value.reshape((1, value.shape[1], 32, 128)).permute(0, 2, 1, 3)
        kv[i] = (key, value)
    return tuple(kv)
p.add_argument("--path", type = str, default = "7k_prompts/1.txt", help="Path to the context file")
p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
# p.add_argument("--kv_path", type = int, default = 0)
p.add_argument("--generate_kv", action="store_true", default = None)
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--vanilla", action="store_true", default = None)
p.add_argument("--doc_id", type=int, default = 0)
p.add_argument("--results_dir", type=str, default = "results")
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--quantization_config", type=str, default="config/quantization.json")

args = p.parse_args()
if __name__ == "__main__":
    # 
    model, tokenizer = load_model(
            args.model_id,
            device="cuda",
            num_gpus=args.num_gpus,
            max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Model and tokenizer loaded")
    # Generate KV cache here 
    # if args.generate_kv:
    with open(args.path, "r") as f:
        text = f.read()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    st = time.monotonic()
    # with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     # with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     # input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    #     generated = model.generate(input_ids, max_new_tokens = 1)
    generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
    torch.cuda.synchronize()
    print( f"TTFT: {time.monotonic() - st}" )
    kv = generated['past_key_values']
    kv = list(kv)
    for i in range(len(kv)):
        kv[i] = list(kv[i])
        kv[i][0] = kv[i][0][:, :, :-1]
        kv[i][1] = kv[i][1][:, :, :-1]
        kv[i] = tuple(kv[i])
    kv = tuple(kv)
    kv = compress(kv, pd.read_csv(('sim_results/chunk_to_q_0.csv')))
    # pickle.dump(kv, open(f"{args.save_dir}/test_kv.pkl", "wb"))
    generated = model.generate(input_ids, past_key_values=kv, max_new_tokens=20)
    print(tokenizer.decode(generated[0][-20:]))