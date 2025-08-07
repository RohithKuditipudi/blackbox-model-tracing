import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, LlamaConfig, LlamaForCausalLM
import wandb
import random
import torch
import numpy as np
import scipy as scp
import time
import shutil
import pickle

from tracing.llm import train_model, generate, evaluate_model
from vllm import SamplingParams

import torch.distributed as dist

def experiment_metric(text, prediction, prompt):
    if len(text) <= len(prompt):
        return 0.0
    return sum(prediction[len(prompt):]).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_partial", type=int, default=1)
    parser.add_argument("--n_base", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--shuffle_partition", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=10)
    parser.add_argument("--include_prompt", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--rerun_partitions", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=512)

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    texts = dataset["train"]["text"]

    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]


    # Shuffle the dataset
    random.seed(args.seed)
    random.shuffle(texts)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size, # 256
        intermediate_size=args.intermediate_size, # 512
        num_hidden_layers=args.num_hidden_layers, # 4
        num_attention_heads=args.num_attention_heads, # 8
        max_position_embeddings=args.max_position_embeddings, # 512
        rms_norm_eps=1e-6,
    )

    # Create save directories
    partial_save_path = os.path.join(args.save_dir, "base_model_partial")
    os.makedirs(partial_save_path, exist_ok=True)
    partial_optimizer_path = os.path.join(partial_save_path, "optimizer.pt")

    partial_base_model_path = os.path.join(partial_save_path, f"epoch-{0}")
    if not os.path.exists(partial_base_model_path):
        # Train base model on shuffled full dataset
        print("Training base model...")
        wandb.init(project="tinystories-training", name=f"base_model")
        partial_base_model, partial_optimizer, _ = train_model(
            texts=texts[:args.n_partial],
            config=config,
            tokenizer=tokenizer,
            save_path=partial_save_path,
            index=args.seed,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=False,
        )
        wandb.finish()
        
        torch.save(partial_optimizer.state_dict(), partial_optimizer_path)

    partial_base_model = LlamaForCausalLM.from_pretrained(partial_base_model_path)
    partial_optimizer = torch.optim.AdamW(partial_base_model.parameters(), lr=1e-5)
    partial_optimizer.load_state_dict(torch.load(partial_optimizer_path))
    
    # Create save directories
    base_save_path = os.path.join(args.save_dir, "base_model")
    os.makedirs(base_save_path, exist_ok=True)
    base_model_path = os.path.join(base_save_path, f"epoch-{0}")
    if not os.path.exists(base_model_path):
        # Train base model on shuffled full dataset
        print("Training base model...")
        wandb.init(project="tinystories-training", name=f"base_model")
        _, _, _ = train_model(
            texts=texts[args.n_partial:args.n_base],
            model=partial_base_model,
            optimizer=partial_optimizer,
            tokenizer=tokenizer,
            save_path=base_save_path,
            index=args.seed,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=False,
        )
        wandb.finish()

    # Sample text from base model using prompts from dataset
    print("Generating samples from base model...")
    
    if args.prompt is None:
        # Get first 100 texts from shuffled test set and truncate each to first 20 tokens
        prompts = [tokenizer.decode(tokenizer.encode(text)[:20]) for text in random.sample(list(dataset["validation"]["text"]), k=args.n_samples)]
    else:
        prompts = [args.prompt] * args.n_samples
    
    # Configure sampling parameters
    sampling_params = {
        "temperature": args.temperature,
    }

    # Generate completions using base model checkpoint
    generated_texts = generate(
        prompts=prompts,
        model_path=base_model_path,
        sampling_params=SamplingParams(**sampling_params),
        seed=args.seed,
    )

    # Save generated texts
    samples_path = os.path.join(args.save_dir, "base_model_samples.txt")
    with open(samples_path, "w") as f:
        for prompt, completion in zip(prompts, generated_texts):
            f.write(f"PROMPT:\n{prompt}\n\nCOMPLETION:\n{completion}\n\n{'='*80}\n\n")
    
    print(f"Saved {len(generated_texts)} samples to {samples_path}")

    # Partition the shuffled dataset
    assert len(texts[args.n_partial:args.n_base]) % args.num_partitions == 0, "Number of texts to partition must be divisible by number of partitions"
    partition_size = len(texts[args.n_partial:args.n_base]) // args.num_partitions

    if args.rerun_partitions:
        for i in range(args.num_partitions):
            partition_save_path = os.path.join(args.save_dir, f"partition_{i}")
            if os.path.exists(partition_save_path):
                shutil.rmtree(partition_save_path)
    
    # Fine-tune on each partition
    for i in range(args.num_partitions):
        print(f"Fine-tuning on partition {i+1}/{args.num_partitions}")
        
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        partition_texts = texts[args.n_partial:args.n_base][start_idx:end_idx]
        
        partition_save_path = os.path.join(args.save_dir, f"partition_{i}")
        os.makedirs(partition_save_path, exist_ok=True)

        if args.fine_tune:
            # Reload so we don't share info across partitions
            partial_base_model = LlamaForCausalLM.from_pretrained(partial_base_model_path)
            partial_optimizer = torch.optim.AdamW(partial_base_model.parameters(), lr=1e-5)
            partial_optimizer.load_state_dict(torch.load(partial_optimizer_path))

            model = partial_base_model
            optimizer = partial_optimizer
        else:
            model = None
            optimizer = None
        
        partition_model_path = os.path.join(partition_save_path, f"epoch-{args.n_epochs-1}")
        if not os.path.exists(partition_model_path):
            wandb.init(project="tinystories-training", name=f"partition_{i}")
            train_model(
                texts=partition_texts,
                config=config,
                tokenizer=tokenizer,
                save_path=partition_save_path,
                index=args.seed + i, # different seed for each partition
                batch_size=args.batch_size,
                epochs=args.n_epochs,
                model=model,
                optimizer=optimizer,
                shuffle=args.shuffle_partition,
                reshuffle=True,
            )
            wandb.finish()
    
    # Evaluate log likelihoods of partition models on base model samples
    print("Evaluating partition models on base model samples...")

    # Load the generated texts
    with open(samples_path, "r") as f:
        content = f.read().split("="*80)
        samples = []
        for entry in content:
            if entry.strip():
                parts = entry.split("COMPLETION:\n")
                prompt = parts[0].split("PROMPT:\n")[1].strip() if "PROMPT:\n" in parts[0] else ""
                completion = parts[1].strip()
                if args.include_prompt:
                    samples.append(prompt + completion)
                else:
                    samples.append(completion)

    # Evaluate each partition model
    partition_metrics = []
    for i in range(args.num_partitions):
        print(f"Evaluating partition model {i+1}/{args.num_partitions}")
        
        # Load partition model path
        partition_model_path = os.path.join(args.save_dir, f"partition_{i}", f"epoch-{args.n_epochs-1}")
        
        # Get predictions using evaluate_model
        partition_model = LlamaForCausalLM.from_pretrained(partition_model_path)
        predictions, metrics = evaluate_model(
            model=partition_model,
            tokenizer=tokenizer,
            texts=samples,
            metric=experiment_metric,
            prompts=prompts,
            batch_size=args.batch_size
        )

        partition_metrics.append(np.mean(metrics))

        pickle.dump(predictions, open(os.path.join(args.save_dir, f"partition_{i}_predictions.pkl"), "wb"))
    
    print("P-VALUE INCOMING:")
    time.sleep(0.1)
    print(".")
    time.sleep(0.1)
    print(".")
    time.sleep(0.1) 
    print(".")
    print(scp.stats.spearmanr(partition_metrics, np.arange(len(partition_metrics))))

    # Save partition metrics
    with open(os.path.join(args.save_dir, "partition_metrics.pkl"), "wb") as f:
        pickle.dump(partition_metrics, f)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
