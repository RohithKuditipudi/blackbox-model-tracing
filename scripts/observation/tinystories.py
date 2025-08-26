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

from tracing.llm import train_model, generate, evaluate_model, load_model_and_optimizer, model_exists
from tracing.utils import get_git_revision_hash, thing_exists_lock, file_exists

import torch.distributed as dist

Z_SCORE_EPS = 1e-6

def get_base_model_path(args):
    training_args = get_training_args(args)

    training_hash = hash_args(training_args)
    base_model_path = os.path.join(args.save_dir, "base_models", training_hash)

    return base_model_path


def get_partial_model_path(args, partial_idx=None):
    training_args = get_training_args(args)

    training_hash = hash_args(training_args)
    if partial_idx is None:
        partial_model_path = os.path.join(args.save_dir, "partial_models", training_hash)
    else:
        partial_model_path = os.path.join(args.save_dir, "partial_models", training_hash, f"partial_model-{partial_idx}")

    return partial_model_path


def get_finetune_model_path(args):
    finetuning_args = get_finetuning_args(args)

    finetuning_hash = hash_args(finetuning_args)
    finetune_model_path = os.path.join(args.save_dir, "finetuned_models", finetuning_hash)

    return finetune_model_path


def get_shuffle_model_path(args, shuffle_idx=None):
    shuffling_args = get_shuffling_args(args)

    shuffling_hash = hash_args(shuffling_args)
    if shuffle_idx is None:
        shuffle_model_path = os.path.join(args.save_dir, "shuffled_models", shuffling_hash)
    else:
        shuffle_model_path = os.path.join(args.save_dir, "shuffled_models", shuffling_hash, f"shuffle_{shuffle_idx}")

    return shuffle_model_path


def get_samples_path(args):
    sampling_args = get_sampling_args(args)

    sampling_hash = hash_args(sampling_args)
    samples_path = os.path.join(args.save_dir, "texts", sampling_hash, "samples.txt")

    return samples_path


def get_experiment_log_path(args):
    experiment_hash = hash_args(args)
    experiment_log_path = os.path.join(args.save_dir, "experiment_logs", experiment_hash, "log.pkl")

    return experiment_log_path


def get_shuffle_metrics_path(args):
    test_args = get_testing_args(args)

    test_hash = hash_args(test_args)
    shuffle_metrics_path = os.path.join(args.save_dir, "metrics", test_hash, "shuffle_metrics.pkl")

    return shuffle_metrics_path


def hash_args(args, length=16):
    return hashlib.sha256(str(args).encode()).hexdigest()[:length]


def get_training_args(args):
    training_args = argparse.Namespace()

    training_args.save_dir = args.save_dir
    training_args.git_hash = args.git_hash
    
    training_args.seed = args.seed

    training_args.batch_size = args.batch_size
    training_args.learning_rate = args.learning_rate

    training_args.n_partial_0 = args.n_partial_0
    training_args.n_base = args.n_base
    training_args.num_partial_models = args.num_partial_models

    training_args.hidden_size = args.hidden_size
    training_args.intermediate_size = args.intermediate_size
    training_args.num_hidden_layers = args.num_hidden_layers
    training_args.num_attention_heads = args.num_attention_heads
    training_args.max_position_embeddings = args.max_position_embeddings
    
    return training_args


def get_finetuning_args(args):
    finetuning_args = argparse.Namespace()

    finetuning_args.save_dir = args.save_dir
    finetuning_args.git_hash = args.git_hash
    
    finetuning_args.seed = args.seed

    finetuning_args.batch_size = args.batch_size
    finetuning_args.learning_rate = args.learning_rate

    finetuning_args.n_base = args.n_base
    finetuning_args.n_finetune = args.n_finetune
    finetuning_args.reinit_ft_optimizer = args.reinit_ft_optimizer

    finetuning_args.base_model_path = args.base_model_path

    return finetuning_args


def get_sampling_args(args):
    sampling_args = argparse.Namespace()

    sampling_args.save_dir = args.save_dir
    sampling_args.git_hash = args.git_hash
    
    sampling_args.sampling_seed = args.sampling_seed

    sampling_args.temperature = args.temperature
    sampling_args.max_tokens = args.max_tokens
    sampling_args.n_sample = args.n_sample

    sampling_args.prompt = args.prompt

    sampling_args.finetune_model_path = args.finetune_model_path

    return sampling_args


def get_shuffling_args(args):
    shuffle_args = argparse.Namespace()

    shuffle_args.save_dir = args.save_dir
    shuffle_args.git_hash = args.git_hash
    
    shuffle_args.seed = args.seed
    shuffle_args.shuffle_seed = args.shuffle_seed

    shuffle_args.batch_size = args.batch_size
    shuffle_args.num_shuffles = args.num_shuffles

    shuffle_args.n_partial_0 = args.n_partial_0
    shuffle_args.n_base = args.n_base
    shuffle_args.num_partial_models = args.num_partial_models
    shuffle_args.partial_model_index = args.partial_model_index

    shuffle_args.partial_model_path = args.partial_model_path

    return shuffle_args


def get_testing_args(args):
    test_args = argparse.Namespace()

    test_args.save_dir = args.save_dir
    test_args.git_hash = args.git_hash

    test_args.batch_size = args.batch_size
    test_args.learning_rate = args.learning_rate

    test_args.ref_model_path = args.ref_model_path
    test_args.finetune_on_test = args.finetune_on_test
    test_args.num_shuffles = args.num_shuffles
    
    test_args.samples_path = args.samples_path
    test_args.shuffle_model_dir = args.shuffle_model_dir

    return test_args


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


def get_sampling_prompts(seed=0, length=20, n_sample=1):
    dataset = get_dataset()
    tokenizer = get_tokenizer()

    random.seed(seed)
    prompts = [
        tokenizer.decode(tokenizer.encode(text)[:length]) 
        for text in random.sample(list(dataset["validation"]["text"]), k=n_sample)
    ]

    return prompts


def read_samples(samples_path):
    # Load the generated texts
    prompts,samples = [],[]
    with open(samples_path, "r") as f:
        content = f.read().split("="*80)
        for entry in content:
            if entry.strip():
                parts = entry.split("COMPLETION:\n")
                prompt = parts[0].split("PROMPT:\n")[1].strip() if "PROMPT:\n" in parts[0] else ""
                completion = parts[1].strip()

                prompts.append(prompt)
                samples.append(prompt + completion)

    return prompts, samples


def write_samples(samples_path, prompts, completions):
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    with open(samples_path, "w") as f:
        for prompt, completion in zip(prompts, completions):
            f.write(f"PROMPT:\n{prompt}\n\nCOMPLETION:\n{completion}\n\n{'='*80}\n\n")


def read_shuffle_metrics(shuffle_metrics_path):
    with open(shuffle_metrics_path, "rb") as f:
        return pickle.load(f)


def write_shuffle_metrics(shuffle_metrics_path, shuffle_metrics):
    os.makedirs(os.path.dirname(shuffle_metrics_path), exist_ok=True)
    with open(shuffle_metrics_path, "wb") as f:
        pickle.dump(shuffle_metrics, f)


def read_experiment_log(experiment_log_path):
    with open(experiment_log_path, "rb") as f:
        return pickle.load(f)


def write_experiment_log(experiment_log_path, experiment_log):
    os.makedirs(os.path.dirname(experiment_log_path), exist_ok=True)
    with open(experiment_log_path, "wb") as f:
        pickle.dump(experiment_log, f)
        

def get_default_optimizer_params(args):
    return {"lr": args.learning_rate}


def get_n_partial(partial_idx, n_partial_0, n_base, num_partial_models):
    assert (n_base - n_partial_0) % (num_partial_models) == 0, "num_partial_models must be a factor of n_base - n_partial_0"
    interval_size = (n_base - n_partial_0) // (num_partial_models)

    return n_partial_0 + partial_idx * interval_size


def run_training(args):
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

    base_model_path = get_base_model_path(args)
    with thing_exists_lock(path=base_model_path, thing_exists_fn=model_exists) as thing_exists:
        if thing_exists:
            print("Base model already exists, skipping training")
        else:
            # Train base model on shuffled full dataset
            print("Training base model...")
            wandb.init(project="tinystories-training", name=f"base_model")
            train_model(
                texts=texts[:args.n_partial_0],
                config=config,
                tokenizer=tokenizer,
                save_path=get_partial_model_path(args, partial_idx=0),
                batch_size=args.batch_size,
                epochs=1,
                shuffle=False,
                optimizer_params=optimizer_params,
            )

            assert (args.n_base - args.n_partial_0) % (args.num_partial_models) == 0, "num_partial_models must be a factor of n_base - n_partial_0"
            interval_size = (args.n_base - args.n_partial_0) // (args.num_partial_models)

            if args.num_partial_models > 1:
                for i in range(1, args.num_partial_models):
                    n_partial = get_n_partial(
                        partial_idx=i, 
                        n_partial_0=args.n_partial_0, 
                        n_base=args.n_base, 
                        num_partial_models=args.num_partial_models
                    )
                    partial_model_path = get_partial_model_path(args, partial_idx=i)
                    prev_partial_model_path = get_partial_model_path(args, partial_idx=i-1)

                    model, optimizer = load_model_and_optimizer(prev_partial_model_path)
                    train_model(
                        texts=texts[:n_partial][-interval_size:],
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        save_path=partial_model_path,
                        batch_size=args.batch_size,
                        epochs=1,
                        shuffle=False,
                    )

            final_partial_model_path = get_partial_model_path(args, partial_idx=args.num_partial_models-1)
            model, optimizer = load_model_and_optimizer(final_partial_model_path)
            train_model(
                texts=texts[:args.n_base][-interval_size:],
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                save_path=base_model_path,
                batch_size=args.batch_size,
                epochs=1,
                shuffle=False,
            )
            wandb.finish()


def run_finetuning(args):
    tokenizer = get_tokenizer()
    texts = get_training_texts(args.seed)

    ft_model_path = get_finetune_model_path(args)
    with thing_exists_lock(path=ft_model_path, thing_exists_fn=model_exists) as thing_exists:
        if thing_exists:
            print("Finetuned model already exists, skipping finetuning")
        else:
            # Fine-tune on shuffled full dataset
            print("Fine-tuning from base model...")
            wandb.init(project="tinystories-training", name=f"finetune")
            model, optimizer = load_model_and_optimizer(args.base_model_path)
            if args.reinit_ft_optimizer:
                optimizer_params = get_default_optimizer_params(args)
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
            train_model(
                texts=texts[args.n_base:args.n_base+args.n_finetune],
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                save_path=ft_model_path,
                batch_size=args.batch_size,
                epochs=1,
                shuffle=False,
            )
            wandb.finish()


def run_testing(args):
    shuffle_metrics_path = get_shuffle_metrics_path(args)

    with thing_exists_lock(path=shuffle_metrics_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Shuffle metrics already exists, skipping to z-score calculation")
        else:
            tokenizer = get_tokenizer()
            prompts, samples = read_samples(
                samples_path=args.samples_path,
            )

            # Evaluate each shuffle model
            shuffle_metrics = []
            for i in range(args.num_shuffles):
                print(f"Evaluating shuffle model {i+1}/{args.num_shuffles}")

                tmp_model_path = os.path.join(args.shuffle_model_dir, f"shuffle_{i}")
                tmp_model, _ = load_model_and_optimizer(tmp_model_path)
                
                if args.finetune_on_test:
                    wandb.init(project="tinystories-training", name=f"tmp_model")
                    tmp_model, _, _ = train_model(
                        texts=samples,
                        model=tmp_model,
                        tokenizer=tokenizer,
                        batch_size=args.batch_size,
                        epochs=1,
                        shuffle=False,
                        optimizer_params=get_default_optimizer_params(args),
                    )
                    wandb.finish()
                    
                _, metrics = evaluate_model(
                    model=tmp_model,
                    tokenizer=tokenizer,
                    texts=samples,
                    metric=experiment_metric,
                    prompts=prompts,
                    batch_size=args.batch_size
                )

                shuffle_metrics.append(np.mean(metrics))
            
            write_shuffle_metrics(shuffle_metrics_path=shuffle_metrics_path, shuffle_metrics=shuffle_metrics)
        
    shuffle_metrics = read_shuffle_metrics(shuffle_metrics_path)
    z_score = (shuffle_metrics[-1] - np.mean(shuffle_metrics[:-1])) / (np.std(shuffle_metrics[:-1]) + Z_SCORE_EPS)

    return z_score, shuffle_metrics


def run_sampling(args):
    # Configure sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.prompt is None:
        prompts = get_sampling_prompts(seed=args.sampling_seed, n_sample=args.n_sample)
    else:
        prompts = [args.prompt] * args.n_sample

    samples_path = get_samples_path(args)
    with thing_exists_lock(path=samples_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Samples already exist, skipping sampling")
        else:
            # Generate completions using model checkpoint
            print("Generating samples from model...")
            completions = generate(
                prompts=prompts,
                model_checkpoint_path=os.path.join(args.finetune_model_path, "epoch-0"),
                sampling_params=SamplingParams(**sampling_params),
                seed=args.sampling_seed,
            )

            # Save generated texts
            write_samples(samples_path=samples_path, prompts=prompts, completions=completions)
        
            print(f"Saved {len(completions)} samples to {samples_path}")
    

def run_shuffling(args):
    tokenizer = get_tokenizer()
    texts = get_training_texts(args.seed)

    n_partial = get_n_partial(
        partial_idx=args.partial_model_index,
        n_partial_0=args.n_partial_0, 
        n_base=args.n_base, 
        num_partial_models=args.num_partial_models
    )

    retrain_texts = texts[n_partial:args.n_base]

    last_shuffle_model_path = get_shuffle_model_path(args, shuffle_idx=args.num_shuffles-1)
    with thing_exists_lock(path=last_shuffle_model_path, thing_exists_fn=model_exists) as thing_exists:
        if thing_exists:
            print("Shuffled models already exist, skipping shuffling")
        else:
            # Fine-tune on each shuffle
            for i in range(args.num_shuffles):
                if i == args.num_shuffles - 1:
                    shuffle = False
                else:
                    shuffle = True

                print(f"Fine-tuning on shuffle {i+1}/{args.num_shuffles}")

                shuffle_model_path = get_shuffle_model_path(args, shuffle_idx=i)
                model, optimizer = load_model_and_optimizer(args.partial_model_path)

                wandb.init(project="tinystories-training", name=f"shuffle_{i}")
                train_model(
                    texts=retrain_texts,
                    tokenizer=tokenizer,
                    save_path=shuffle_model_path,
                    index=args.shuffle_seed + i, # different seed for each model
                    batch_size=args.batch_size,
                    epochs=1,
                    model=model,
                    optimizer=optimizer,
                    shuffle=shuffle,
                )
                wandb.finish()


def experiment_metric(tokenized_text, prediction, tokenized_prompt):
    if len(tokenized_text) <= len(tokenized_prompt):
        return 0.0
    return sum(prediction[len(tokenized_prompt):]).item()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_partial_0", type=int, default=0)
    parser.add_argument("--num_partial_models", type=int, default=1)
    parser.add_argument("--n_base", type=int, default=1)
    parser.add_argument("--n_finetune", type=int, default=0)
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--sampling_seed", type=int, default=0)
    parser.add_argument("--num_shuffles", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--finetune_on_test", action="store_true", default=False)
    parser.add_argument("--reinit_ft_optimizer", action="store_true", default=False)
    parser.add_argument("--partial_model_index", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--include_hash", action="store_true", default=False)

    args = parser.parse_args()

    args.git_hash = get_git_revision_hash() if args.include_hash else None

    args.base_model_path = get_base_model_path(args)
    args.finetune_model_path = get_finetune_model_path(args)
    args.partial_model_path = get_partial_model_path(args, args.partial_model_index)
    args.samples_path = get_samples_path(args)
    args.shuffle_model_dir = get_shuffle_model_path(args)
    
    training_args = get_training_args(args)
    run_training(training_args)

    finetuning_args = get_finetuning_args(args)
    run_finetuning(finetuning_args)

    sampling_args = get_sampling_args(args)
    run_sampling(sampling_args)

    shuffling_args = get_shuffling_args(args)
    run_shuffling(shuffling_args)

    testing_args = get_testing_args(args)
    z_score, shuffle_metrics = run_testing(testing_args)

    experiment_log = {"z_score": z_score, "metrics": shuffle_metrics, "args": args}
    experiment_log_path = get_experiment_log_path(args)
    with thing_exists_lock(path=experiment_log_path, thing_exists_fn=file_exists) as thing_exists:
        if not thing_exists:
            write_experiment_log(experiment_log_path=experiment_log_path, experiment_log=experiment_log)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()