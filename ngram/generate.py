import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import evaluate
# import wandb
from datasets import load_dataset

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import argparse
import json
import hashlib
import random
import subprocess
import csv
import pickle 

def save_texts_pickle(prompts, filename):
    with open(filename, 'wb') as f:
        pickle.dump(prompts, f)

def truncate_tokenize_prompts(prompts, tokenizer, max_tokens=16):
    # Load the tokenizer
    
    truncated_prompts = []
    
    for prompt in tqdm(prompts, desc="Truncating prompts"):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        truncated_tokens = tokens[:max_tokens]
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        truncated_prompts.append(truncated_prompt)
    
    return truncated_prompts

def generate_synthetic_data_prompts(model_path, revision, tokenizer, prompts, temperature=0.8):
    """Generate synthetic text data using vLLM"""
    llm = LLM(model=model_path, revision=revision)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=128)

    truncated_prompts = truncate_tokenize_prompts(prompts, tokenizer)
    outputs = llm.generate(truncated_prompts, sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]

    del outputs, llm 
    torch.cuda.empty_cache()
    return generated_texts

def generate_synthetic_data_scratch(model_path, revision, n_samples, prompt=' ', temperature=0.8):
    """Generate synthetic text data using vLLM"""
    llm = LLM(model=model_path,revision=revision)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=128)

    prompts = [prompt] * n_samples
    outputs = llm.generate(prompts, sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]

    del outputs, llm 
    torch.cuda.empty_cache()
    return generated_texts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='EleutherAI/pythia-6.9b-deduped')
    parser.add_argument("--revision", type=str, default='step100000')
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default='/nlp/scr/salzhu/pythia_datasets')
    parser.add_argument("--save_file", type=str, default='test')
    parser.add_argument("--prompt_dataset", type=str, default="timaeus/dsir-pile-10m")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--scratch', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    texts = None 
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.scratch:
        texts = generate_synthetic_data_scratch(args.model, args.revision,
                                    args.n_samples, temperature=args.temperature)
    else:
        raw_dataset = load_dataset(args.prompt_dataset,split='train')
        prompts = list(raw_dataset['contents'])[args.index * args.n_samples : (args.index + 1) * args.n_samples]
        texts = generate_synthetic_data_prompts(args.model, args.revision, tokenizer,
                                    prompts, temperature=args.temperature)
        del raw_dataset

    file_name = f'{args.save_dir}/{args.save_file}'
    
    save_texts_pickle(texts, f'{args.save_dir}/{args.save_file}_{args.index}.pkl')
    del texts 
    print('done',flush=True)
