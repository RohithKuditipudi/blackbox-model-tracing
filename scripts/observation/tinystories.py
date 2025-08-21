import os
import argparse
from datasets import load_dataset
from sympy.printing.latex import true
from transformers import AutoTokenizer, AutoConfig, LlamaConfig, LlamaForCausalLM
import wandb
import random
import torch
import numpy as np
import scipy as scp
import time
import shutil
import pickle
import subprocess
import hashlib

from tracing.llm import train_model, generate, evaluate_model, load_model_and_optimizer, model_exists
from vllm import SamplingParams

import torch.distributed as dist

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def generate_and_evaluate_samples(base_model_path, tokenizer, prompts, args):
    """Generate samples from base model and evaluate shuffle models"""
    
    # Configure sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Generate completions using base model checkpoint
    print("Generating samples from base model...")
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
    
    # Evaluate log likelihoods of shuffle models on base model samples
    print("Evaluating shuffle models on base model samples...")

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

    # Evaluate each shuffle model
    shuffle_metrics = []
    for i in range(args.num_shuffles):
        print(f"Evaluating shuffle model {i+1}/{args.num_shuffles}")
        
        # Load shuffle model path
        tmp_model_path = os.path.join(args.save_dir, f"shuffle_{i}", f"epoch-{args.n_epochs-1}")
        
        # Get predictions using evaluate_model
        tmp_model = LlamaForCausalLM.from_pretrained(tmp_model_path)
        if args.finetune_on_test:
            wandb.init(project="tinystories-training", name=f"tmp_model")
            tmp_model, _, _ = train_model(
                texts=samples,
                model=tmp_model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                epochs=1,
                shuffle=False,
                optimizer_params={"lr": args.learning_rate},
            )
            wandb.finish()
            
        predictions, metrics = evaluate_model(
            model=tmp_model,
            tokenizer=tokenizer,
            texts=samples,
            metric=experiment_metric,
            prompts=prompts,
            batch_size=args.batch_size
        )

        shuffle_metrics.append(np.mean(metrics))
        pickle.dump(predictions, open(os.path.join(args.save_dir, f"shuffle_{i}_predictions.pkl"), "wb"))
    
    return shuffle_metrics, samples_path

def experiment_metric(text, prediction, prompt):
    if len(text) <= len(prompt):
        return 0.0
    return sum(prediction[len(prompt):]).item()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_partial", type=int, default=1)
    parser.add_argument("--n_partial_checkpoints", type=int, default=0)
    parser.add_argument("--n_retrain", type=int, default=1)
    parser.add_argument("--n_finetune", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shuffles", type=int, default=10)
    parser.add_argument("--include_prompt", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--rerun_shuffles", action="store_true", default=False)
    parser.add_argument("--rerun_finetune", action="store_true", default=False)
    parser.add_argument("--finetune_on_test", action="store_true", default=False)
    parser.add_argument("--reinit_ft_optimizer", action="store_true", default=False)
    parser.add_argument("--checkpoint_index", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    args.n_base = args.n_partial + args.n_retrain

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
    optimizer_params = {"lr": args.learning_rate}

    # Create save directories
    partial_save_path = os.path.join(args.save_dir, "base_model_partial")
    os.makedirs(partial_save_path, exist_ok=True)

    if not model_exists(partial_save_path):
        # Train base model on shuffled full dataset
        print("Training base model...")
        wandb.init(project="tinystories-training", name=f"base_model")
        train_model(
            texts=texts[:args.n_partial],
            config=config,
            tokenizer=tokenizer,
            save_path=partial_save_path,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=False,
            optimizer_params=optimizer_params,
        )
        wandb.finish()

    assert (args.n_base - args.n_partial) % (args.n_partial_checkpoints + 1) == 0, "n_partial_checkpoints must be a factor of n_base - n_partial"
    ckpt_size = (args.n_base - args.n_partial) // (args.n_partial_checkpoints + 1)
    ckpt_save_paths = [partial_save_path]

    if args.n_partial_checkpoints > 0:
        for i in range(args.n_partial_checkpoints):
            ckpt_save_path = os.path.join(partial_save_path, f"ckpt-{i}")
            os.makedirs(ckpt_save_path, exist_ok=True)
            if not model_exists(ckpt_save_path):
                wandb.init(project="tinystories-training", name=f"ckpt-{i}")
                model, optimizer = load_model_and_optimizer(ckpt_save_paths[-1])
                train_model(
                    texts=texts[args.n_partial+i*ckpt_size:args.n_partial+(i+1)*ckpt_size],
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    save_path=ckpt_save_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                )
                wandb.finish()
            
            ckpt_save_paths.append(ckpt_save_path)
                
    
    # Create save directories
    base_save_path = os.path.join(args.save_dir, "base_model")
    os.makedirs(base_save_path, exist_ok=True)

    if not model_exists(base_save_path):
        # Train base model on shuffled full dataset
        print("Training base model...")
        wandb.init(project="tinystories-training", name=f"base_model")
        model, optimizer = load_model_and_optimizer(ckpt_save_paths[-1])
        train_model(
            texts=texts[args.n_partial+args.n_partial_checkpoints*ckpt_size:args.n_base],
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            save_path=base_save_path,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=False,
        )
        wandb.finish()
    
    ft_save_path = os.path.join(args.save_dir, "finetune")
    os.makedirs(ft_save_path, exist_ok=True)
    if not model_exists(ft_save_path) or args.rerun_finetune:
        # Fine-tune on shuffled full dataset
        print("Fine-tuning on shuffled full dataset...")
        wandb.init(project="tinystories-training", name=f"finetune")
        model, optimizer = load_model_and_optimizer(base_save_path)
        if args.reinit_ft_optimizer:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        train_model(
            texts=texts[args.n_base:args.n_base+args.n_finetune],
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            save_path=ft_save_path,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=False,
        )
        wandb.finish()

    if args.rerun_shuffles:
        for i in range(args.num_shuffles):
            shuffle_save_path = os.path.join(args.save_dir, f"shuffle_{i}")
            if os.path.exists(shuffle_save_path):
                shutil.rmtree(shuffle_save_path)
    
    retrain_texts = texts[args.n_partial:args.n_base][:args.n_retrain]
    
    # Fine-tune on each shuffle
    for i in range(args.num_shuffles):
        print(f"Fine-tuning on shuffle {i+1}/{args.num_shuffles}")
        
        shuffle_save_path = os.path.join(args.save_dir, f"shuffle_{i}")
        os.makedirs(shuffle_save_path, exist_ok=True)
        
        if i == args.num_shuffles - 1:
            shuffle_partition = False
        else:
            shuffle_partition = True
            
        if not model_exists(shuffle_save_path, args.n_epochs-1):
            wandb.init(project="tinystories-training", name=f"shuffle_{i}")
            model, optimizer = load_model_and_optimizer(ckpt_save_paths[args.checkpoint_index])
            train_model(
                texts=retrain_texts,
                config=config,
                tokenizer=tokenizer,
                save_path=shuffle_save_path,
                index=args.seed + i, # different seed for each model
                batch_size=args.batch_size,
                epochs=args.n_epochs,
                model=model,
                optimizer=optimizer,
                shuffle=shuffle_partition,
            )
            wandb.finish()
    
    # Sample text from base model using prompts from dataset
    print("Generating samples from base model...")
    
    if args.prompt is None:
        # Get first 100 texts from shuffled test set and truncate each to first 20 tokens
        prompts = [tokenizer.decode(tokenizer.encode(text)[:20]) for text in random.sample(list(dataset["validation"]["text"]), k=args.n_samples)]
    else:
        prompts = [args.prompt] * args.n_samples
    
    ft_model_path = os.path.join(ft_save_path, f"epoch-{0}")
    shuffle_metrics, _ = generate_and_evaluate_samples(ft_model_path, tokenizer, prompts, args)
    if args.ref_model_path is not None:
        ref_shuffle_metrics, _ = generate_and_evaluate_samples(args.ref_model_path, tokenizer, prompts, args)
        slope, intercept, _, _, _ = scp.stats.linregress(ref_shuffle_metrics, shuffle_metrics)

        # Subtract off the best fit line
        shuffle_metrics = np.array(shuffle_metrics) - (slope * np.array(ref_shuffle_metrics) + intercept)
    
    print("Z-SCORE INCOMING:")
    time.sleep(0.1)
    print(".")
    time.sleep(0.1)
    print(".")
    time.sleep(0.1) 
    print(".")

    z_score = (shuffle_metrics[-1] - np.mean(shuffle_metrics[:-1]))/np.std(shuffle_metrics[:-1])
    print(z_score)

    experiment_log = {"z_score": z_score, "metrics": shuffle_metrics, "args": args, "git_hash": get_git_revision_hash()}
    experiment_id = hashlib.sha256(str(experiment_log).encode()).hexdigest()[:16]

    # Save shuffle metrics
    with open(os.path.join(args.log_dir, f"experiment_{experiment_id}.pkl"), "wb") as f:
        pickle.dump(experiment_log, f)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()