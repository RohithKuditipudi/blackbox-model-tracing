import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from vllm import SamplingParams
import wandb
import random
import torch
import numpy as np
import pickle
import hashlib

from tracing.llm import train_model, model_exists
from tracing.utils import thing_exists_lock


def get_dataset():
    dataset = load_dataset("roneneldan/TinyStories")

    return dataset


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_training_texts(seed=0):
    dataset = get_dataset()

    # Pepare training texts
    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]

    # Shuffle training texts
    random.seed(seed)
    random.shuffle(texts)

    return texts


def save_args(args, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(os.path.join(model_path, "args.pkl"), "wb") as f:
        pickle.dump(args, f)


def get_default_optimizer_params(args):
    return {"lr": args.learning_rate}


def run_training(args):
    with thing_exists_lock(path=args.model_path, thing_exists_fn=model_exists) as thing_exists:
        if thing_exists:
            print("Model already exists, skipping training")
        else:
            # Train base model on shuffled full dataset
            tokenizer = get_tokenizer()
            texts = get_training_texts(seed=args.seed)

            config = LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=args.hidden_size, # 256
                intermediate_size=args.intermediate_size, # 512
                num_hidden_layers=args.num_hidden_layers, # 4
                num_attention_heads=args.num_attention_heads, # 8
                max_position_embeddings=args.max_position_embeddings, # 512
                rms_norm_eps=1e-6,
            )
            optimizer_params = get_default_optimizer_params(args)
            
            # Train base model on shuffled full dataset
            print("Training base model...")
            wandb.init(project="tinystories-training", name=f"null_model")
            train_model(
                texts=texts[:args.n_train],
                config=config,
                tokenizer=tokenizer,
                save_path=args.model_path,
                batch_size=args.batch_size,
                epochs=1,
                shuffle=False,
                optimizer_params=optimizer_params,
            )
            wandb.finish()
            save_args(args, args.model_path)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    training_args = parser.parse_args()
    
    run_training(training_args)


if __name__ == "__main__":
    main()