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
import argparse

def train_tiny(texts, config, tokenizer, save_dir, df, index, batch_size=1, epochs=1):
    model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(epochs):
        shuffle_order = list(range(len(texts)))
        random.shuffle(shuffle_order)
        df[f'order-{index}-epoch-{epoch}'] = shuffle_order
        shuffled_texts = [texts[i] for i in shuffle_order]

        train_dataloader = DataLoader(shuffled_texts, batch_size=batch_size, shuffle=False) # assume train_texts is shuffled in desired order
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

        model.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}'))
        
        pplx = eval_tiny(os.path.join(save_dir, f'epoch-{epoch}'), texts)
        df[f'pplx-{index}-epoch-{epoch}'] = pplx

def eval_tiny(model_path, eval_texts):
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=eval_texts)
    pplx = np.log(result['perplexities'])

    return pplx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--n_train_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()

    N = args.n
    N_TRAIN_SAMPLES = args.n_train_samples

    INDEX = args.index
    random.seed(INDEX)

    SAVE_DIR = args.save_dir
    REF_PATH = os.path.join(SAVE_DIR, f'tiny_ref_model_{INDEX}')
    DF_PATH = os.path.join(REF_PATH, f'tinystories_{INDEX}.csv')

    df = pd.DataFrame({})

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

    texts = dataset["train"]["text"][:N_TRAIN_SAMPLES]
    texts = [item for item in texts if item != ""]

    for i in range(N):

        print(f"Training model {i}...")
    
        train_tiny(texts, config, tokenizer, REF_PATH, df, i, batch_size=args.batch_size, epochs=args.epochs)
        df.to_csv(DF_PATH)
