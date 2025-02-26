import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import random
import numpy as np
import evaluate
import pandas as pd
import os
import argparse

from train_toy_models import train_tiny, eval_tiny 

batch_sizes = [64, 32, 16, 8, 4, 2, 1]

ORDER_PATH = '/nlp/u/rohithk/blackbox-model-tracing/results/tiny_ref_pplx_2.csv' # use the ordering from here
REF_PATH = '/nlp/u/salzhu/blackbox-model-tracing/tinystories_rohith/batch_ablations'
DF_PATH = '/nlp/u/salzhu/blackbox-model-tracing/tinystories_rohith/bs_pplx.csv'
N_TRAIN_SAMPLES = 100000
N_EVAL_SAMPLES = 10000

df_order = pd.read_csv(ORDER_PATH)
ordering = df_order['order']

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = tokenizer.eos_token

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)

# train everything with the same fixed order
texts = dataset["train"]["text"][:N_TRAIN_SAMPLES]
texts = [item for item in texts if item != ""] # some sequences are bad
texts = [texts[i] for i in ordering]

del dataset 

df = pd.DataFrame({})

for bs in batch_sizes:
    print(f'Training batch size {bs}...')
    save_path = os.path.join(REF_PATH, f'bs_{bs}')
    train_tiny(texts, config, tokenizer, save_path, bs)
    pplx = eval_tiny(os.path.join(save_path, 'final'), texts[:N_EVAL_SAMPLES])
    df[f'pplx-bs-{bs}'] = pplx
    df.to_csv(DF_PATH)
