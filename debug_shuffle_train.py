# I ran: python this_script 34

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import random
import numpy as np
import evaluate
import pandas as pd
from tqdm import tqdm
import os
import sys

N = 4
N_TRAIN_SAMPLES = 10000

def train_tiny(train_texts, config, tokenizer, save_dir):
    model = LlamaForCausalLM(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_dataloader = DataLoader(train_texts, batch_size=1, shuffle=False) # assume train_texts is shuffled in desired order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    batch_iterator = tqdm(train_dataloader)

    for batch in batch_iterator:
        
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        inputs['labels'] = inputs['input_ids'].clone()

        outputs = model(**inputs)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if save_dir is not None:
        model.save_pretrained(save_dir)

def eval(model_path, eval_texts):
    perplexity = evaluate.load("perplexity", module_type="metric")
    eval = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=eval_texts)
    pplx = np.log(eval['perplexities'])

    return pplx

if __name__ == "__main__":

    INDEX = sys.argv[1] 
    random.seed(INDEX)

    REF_PATH = f'/nlp/u/rohithk/blackbox-model-tracing/train_references/debug/tiny_ref_model_{INDEX}'
    DF_PATH = f'/nlp/u/rohithk/blackbox-model-tracing/train_references/debug/tinystories_refmodels_{INDEX}.csv'

    if os.path.exists(DF_PATH):
        df = pd.read_csv(DF_PATH)
    else:
        df = pd.DataFrame({})

    dataset = load_dataset("roneneldan/TinyStories")

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(REF_PATH)

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )

    texts = dataset["train"]["text"][:N_TRAIN_SAMPLES]
    random.shuffle(texts) # CHANGE: shuffled here
    texts = [item for item in texts if item != ""]

    for i in range(N):

        print(f"Training model {i}...")

        shuffle_order = list(range(len(texts)))
        random.shuffle(shuffle_order)
        df[f'order-{i}'] = shuffle_order

        shuffled_texts = [texts[i] for i in shuffle_order]
    
        train_tiny(shuffled_texts, config, tokenizer, REF_PATH)
        pplx = eval(REF_PATH, texts)
        df[f'pplx-{i}'] = pplx

        df.to_csv(DF_PATH)
