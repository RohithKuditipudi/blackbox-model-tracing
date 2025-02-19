"""
Trains a bunch of toy models with different data orderings on the same dataset (TinyStories)
N = number of models to train, and can change N_TRAIN_SAMPLES. 
Can change the model architecture with config @ Line 81. 
Can change dataset @ Line 76.

For N models,
- Shuffles the dataset for training
- Calls "train_tiny" which takes in texts, model config, and place to save (need to pass for eval to work). Trains the model for one epoch.
- Calls "eval", which takes the model save path and evaluates PPLX on given texts (in this case, the eval texts are the same as the train texts)

For each model, PPLX on all sequences, and the data ordering are saved at DF_PATH. 

Main function takes in a seed as input
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import random
import numpy as np
import evaluate
import pandas as pd
import os
import sys

N = 20
N_TRAIN_SAMPLES = 100000
EPOCHS = 10

def train_tiny(train_texts, config, tokenizer, save_dir):
    model = LlamaForCausalLM(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_dataloader = DataLoader(train_texts, batch_size=100, shuffle=False) # assume train_texts is shuffled in desired order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in train_dataloader:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inputs['labels'] = inputs['input_ids'].clone()

            outputs = model(**inputs)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {loss.item():.4f}")

        # Save checkpoint after each epoch
        if save_dir is not None:
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)  # Save tokenizer with each checkpoint
            print(f"Saved checkpoint to {epoch_dir}")

    # Save final model
    if save_dir is not None:
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

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

    REF_PATH = f'/nlp/u/rohithk/blackbox-model-tracing/results/tiny_ref_model_{INDEX}'
    DF_PATH = f'/nlp/u/rohithk/blackbox-model-tracing/results/tiny_ref_pplx_{INDEX}.csv'

    if os.path.exists(DF_PATH):
        df = pd.read_csv(DF_PATH)
    else:
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
    texts = [item for item in texts if item != ""] # some sequences are bad

    for i in range(N):

        print(f"Training model {i}...")

        shuffle_order = list(range(len(texts)))
        random.shuffle(shuffle_order)
        df[f'order-{i}'] = shuffle_order

        shuffled_texts = [texts[i] for i in shuffle_order]
    
        train_tiny(shuffled_texts, config, tokenizer, REF_PATH)
        pplx = eval(os.path.join(REF_PATH, "final"), texts)
        df[f'pplx-{i}'] = pplx
        df[f'order-{i}'] = shuffle_order

        df.to_csv(DF_PATH)
