# I ran: python this_script 34

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import evaluate
import wandb

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import argparse
import json
import hashlib
import random
import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

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

        for batch_idx, batch in enumerate(batch_iterator): 
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inputs['labels'] = inputs['input_ids'].clone()

            outputs = model(**inputs)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "epoch": epoch,
            })

        model.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'))
        
        pplx = eval_tiny(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'), texts)
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

    args_dict = vars(args)
    args_dict['git_commit'] = get_git_revision_hash()
    args_str = json.dumps(args_dict, indent=2)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]

    N = args.n
    N_TRAIN_SAMPLES = args.n_train_samples

    INDEX = args.index
    random.seed(INDEX)

    SAVE_DIR = args.save_dir
    REF_PATH = os.path.join(SAVE_DIR, f'tiny_ref_model_{args_hash}')
    DF_PATH = os.path.join(REF_PATH, f'tinystories.csv')

    os.makedirs(REF_PATH, exist_ok=True)
    
    with open(os.path.join(REF_PATH, 'args.json'), 'w') as f:
        f.write(args_str)

    wandb.init(
        project="blackbox-model-tracing",
        config=args_dict,
        name=f"debug_shuffle_train_{args_hash}",
    )

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

    for run_index in range(N):

        print(f"Training model {run_index}...")
    
        train_tiny(texts, config, tokenizer, REF_PATH, df, run_index, batch_size=args.batch_size, epochs=args.epochs)
        df.to_csv(DF_PATH)
