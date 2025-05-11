import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import evaluate

import numpy as np
import pandas as pd
from tqdm import tqdm

import vllm
from vllm import LLM, SamplingParams

import pickle
import random 
import scipy
import copy
from scipy import stats

# from rohith
def correlate(tokens,kgram_index,k_max,shuffle_order):
    shuffled_idx = np.argsort(shuffle_order)
    results = []
    for pos in range(len(tokens)-k_max):
        k = k_max
        while k > 0:
            prefix = tuple(tokens[pos:pos+k])
            if prefix in kgram_index:
                actual_next_token = tokens[pos+k]
                shuffled_idx_avg = 0
                num_matches = 0
                for info in kgram_index[prefix]:
                    if info['next_token'] == actual_next_token:
                        num_matches += 1
                        shuffled_idx_avg += shuffled_idx[info['idx']]
                        
                if num_matches > 0:
                    shuffled_idx_avg /= num_matches
                    results.append(shuffled_idx_avg)
                    break
            k -= 1
                    
    return results

# from rohith
def generate_and_correlate_text(prompts,llm,kgram_index,k,shuffle_order,sampling_params):
    generated = llm.generate(prompts,sampling_params)
    full_results = []
    for i in range(len(generated)):
        tokens = generated[i].outputs[0].token_ids
        results = correlate(tokens,kgram_index,k,shuffle_order)
        
        full_results += results
        
    return full_results

# computes a z-score: shuffles the training order 'reps' times and takes a z-score over the means 
def generate_and_correlate_text_zscore(prompts,llm,kgram_index,k,shuffle_order,sampling_params, reps):
    shuffle_order = copy.deepcopy(shuffle_order)
    generated = llm.generate(prompts,sampling_params) # uncomment if loading from the pickle
    full_results_unshuffled = []
    for i in range(len(generated)):
        tokens = generated[i].outputs[0].token_ids
        results = correlate(tokens,kgram_index,k,shuffle_order)
        
        full_results_unshuffled += results
        
    mean_unshuffled = np.mean(full_results_unshuffled)
    print(mean_unshuffled)

    means = []
    
    for j in range(reps):
        temp = []
        random.shuffle(shuffle_order)
        for i in range(len(generated)):
        
            tokens = generated[i].outputs[0].token_ids
            results = correlate(tokens,kgram_index,k,shuffle_order)
            
            temp += results
        means.append(np.mean(temp))
        print(np.mean(temp))
    
    z_score = (mean_unshuffled - np.mean(means)) / np.std(means)
    p_value = scipy.stats.norm.pdf(abs(z_score))
    print(p_value)
    return mean_unshuffled, means

# computes the kolmogorov smirnov test, comparing the unshuffled indices with 'reps' shuffled indices
def generate_and_correlate_text_ksdist(prompts,llm,kgram_index,k,shuffle_order,sampling_params, reps):
    shuffle_order = copy.deepcopy(shuffle_order)
    generated = llm.generate(prompts,sampling_params) # uncomment if loading from the pickle
    full_results_unshuffled = []
    for i in range(len(generated)):
        tokens = generated[i].outputs[0].token_ids
        results = correlate(tokens,kgram_index,k,shuffle_order)
        
        full_results_unshuffled += results
        
    print(full_results_unshuffled)
    full_others = []
    
    for j in range(reps):
        temp = []
        random.shuffle(shuffle_order)
        for i in range(len(generated)):
        
            tokens = generated[i].outputs[0].token_ids
            results = correlate(tokens,kgram_index,k,shuffle_order)
            
            temp += results
        full_others.append(temp)

    print(full_others)

    # concatenate all the shuffled distributions --- maybe shouldn't do this
    print(f"original: {stats.ks_2samp(full_results_unshuffled, sum(full_others, []))}")

    for j in range(reps):
        print(f"shuffled {j}: {stats.ks_2samp(full_results_unshuffled, full_others[j])}")
