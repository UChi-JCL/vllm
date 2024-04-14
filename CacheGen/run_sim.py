import torch
import os
import time
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM  

all_quantization_ratio = [0.215, 0.277, 0.31]
ratio_to_config = {0.215: "2", 0.277: "1", 0.31: "3"}
baseline_size = 1024 * 120/1e6
def get_comp_time(current_kv_length, prompt_length):
    """ Given the length of KV cache that has already be loaded, predict the 
    delay of computing the next KV chunk using text of length prompt_length
    
    """
    return 1.3
    # if current_kv_length == 0:
    #     input_ids = torch.randint(0, 32000, (1, prompt_length)).cuda()
    #     start = time.monotonic()
    #     model.generate(input_ids, 
    #                    do_sample=False, 
    #                    max_length=prompt_length + 1,
    #                    )
    #     torch.cuda.synchronize()
    #     end = time.monotonic()
    #     return end - start
    
    # input_ids = torch.randint(0, 32000, (1, current_kv_length)).cuda()
    
    # dummy_past_kv = model.generate(input_ids, 
    #                                do_sample=False, 
    #                                max_length=current_kv_length+1,
    #                                 return_dict_in_generate=True
    #                                )['past_key_values']
    # start = time.monotonic()
    # input_ids = torch.randint(0, 32000, (1, prompt_length + current_kv_length)).cuda()
    # model.generate(input_ids, 
    #                do_sample=False, 
    #                max_length=current_kv_length+prompt_length + 1,
    #                past_key_values=dummy_past_kv,
    #                )
    # torch.cuda.synchronize()
    # end = time.monotonic()    
    # breakpoint()
    # return end - start
    
def estimate_delay_best_quantization_quality(chunk_length, throughput, SLO):
    """ Given the chunk_id and SLO, find the best quantization quality that 
    satisfies the SLO
    
    """
    all_ttfts = []
    
    for quantization_ratio in all_quantization_ratio:
        current_kv_size = int(quantization_ratio * chunk_length * baseline_size) # in MB
        ttft = current_kv_size * 8 / (throughput * 1000) 
        all_ttfts.append(ttft)
    # print(all_ttfts)
    for i, ttft in enumerate(all_ttfts[::-1]):
        if ttft < SLO:
            return ttft, all_quantization_ratio[::-1][i]
    return None, None

def get_next_quality(current_kvs, chunk_id, throughput, SLO, chunk_length):
    text_ttft = get_comp_time(current_kvs, chunk_length)
    cachegen_ttft, cachegen_config = estimate_delay_best_quantization_quality(chunk_length, throughput, SLO)
    if cachegen_ttft is None:
        return True, text_ttft, None, None
    if text_ttft <= cachegen_ttft:
        return True, text_ttft, None, None
    elif text_ttft > cachegen_ttft:
        return False, text_ttft, cachegen_ttft, cachegen_config
    
    
def bw_generator(num_chunks):
    import numpy as np
    import random
    min = 0.2
    max = 10
    bw = np.zeros(num_chunks)
    for i in range(num_chunks):
        bw[i] = random.uniform(min, max)
    return bw


context_length = 9600
chunk_size = 1500
all_chunk_sizes = [chunk_size] * (context_length // chunk_size)
all_chunk_sizes += [context_length % chunk_size]
for i in range(len(all_chunk_sizes)):
    all_chunk_sizes[i] += 1
# all_bws = [2] * len(all_chunk_sizes)
# all_bws[3] = 0.5
# all_bws[4] = 0.5
all_bws = bw_generator(10)
# all_bws = [2, 2, 2, 0.2, 0.2, 1, 1, 1, 1, 1]
print(all_bws)
for _ in range(1):
    SLO = 3.3/7
    total_ttft = 0
    df = {"chunk_id": [],  "quality": []}
    df_no_adapt = {}
    for quant_ratio in all_quantization_ratio:
        df_no_adapt[quant_ratio] = 0
    df_baseline = {}
    for quant_ratio in [0.375, 0.5, 1]:
        df_baseline[quant_ratio] = 0
    
    for i in range(len(all_chunk_sizes)):
        
        # print(i * chunk_size, i, all_bws[i], 1, all_chunk_sizes[i])
        if i == 0:
            BW = all_bws[i]
        else:
            BW =  all_bws[i-1]
        is_text, text_ttft, cachegen_ttft, cachegen_config = get_next_quality(i * chunk_size, i, BW, SLO, all_chunk_sizes[i])
        print(BW, is_text, text_ttft, cachegen_ttft, all_chunk_sizes[i], cachegen_config)
        if is_text:
            total_ttft += text_ttft
        else:
            total_ttft += cachegen_ttft
        if is_text:
            df["chunk_id"] += [i]
            df["quality"] += ["0"]
        else:
            df["chunk_id"] += [i]
            df["quality"] += [ratio_to_config[cachegen_config]]
        for quant_ratio in df_no_adapt:
            df_no_adapt[quant_ratio] += all_chunk_sizes[i] * baseline_size * quant_ratio* 8 / (all_bws[i] * 1000)
        for quant_ratio in df_baseline:
            df_baseline[quant_ratio] += all_chunk_sizes[i] * baseline_size * quant_ratio* 8 / (all_bws[i] * 1000)
    df = pd.DataFrame(df)
    df.to_csv(f"sim_results/chunk_to_q_{os.environ['TRACE']}.csv")
    pickle.dump(df_no_adapt, open(f"sim_results/no_adapt_{os.environ['TRACE']}.pkl", "wb"))
    pickle.dump(df_baseline, open(f"sim_results/baseline_{os.environ['TRACE']}.pkl", "wb"))
    
    print(total_ttft)
    
    
    
