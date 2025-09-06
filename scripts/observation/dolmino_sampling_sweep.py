import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
import hashlib
import pickle
import itertools

from tracing.llm import evaluate_model
from tracing.utils import thing_exists_lock, file_exists

import torch.distributed as dist

Z_SCORE_EPS = 1e-6

MODEL_NAME_DICT = {
    "1B": "allenai/OLMo-2-0425-1B",
    "7B": "allenai/OLMo-2-1124-7B",
    "13B": "allenai/OLMo-2-1124-13B",
}
REVISION_TEMPLATE_DICT = {
    "1B": "stage2-ingredient{revision_id}-step23852-tokens51B",
    "7B": "stage2-ingredient{revision_id}-step11931-tokens50B",
    "13B": "stage2-ingredient{revision_id}-step11931-tokens100B",
}


def hash_args(args, length=16):
    return hashlib.sha256(str(args).encode()).hexdigest()[:length]


def get_experiment_log_path(args):
    experiment_hash = hash_args(args)
    experiment_log_path = os.path.join(args.save_dir, "experiment_logs", experiment_hash, "log.pkl")

    return experiment_log_path


def write_experiment_log(experiment_log_path, experiment_log):
    os.makedirs(os.path.dirname(experiment_log_path), exist_ok=True)
    with open(experiment_log_path, "wb") as f:
        pickle.dump(experiment_log, f)


def generate(prompts, model, tokenizer, sampling_params, prompt_template="{prompt}"):

    # Ensure a pad token exists for padding (common for decoder-only LMs)
    if tokenizer.pad_token is None:
        # fall back to eos as pad; common HF practice for causal LMs
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]

    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        return_token_type_ids=False,
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to('cuda') for k,v in inputs.items()}
    # Track per-item input lengths so we can slice completions only
    input_lengths = (inputs["attention_mask"].sum(dim=1)).tolist()
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            max_new_tokens=sampling_params["max_tokens"],
            do_sample=True, 
            temperature=sampling_params["temperature"]
        )
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    completions = [text[length:] for text, length in zip(generated_texts, input_lengths)]
    
    return completions


def get_samples_path(args):
    samples_path = os.path.join(args.save_dir, "samples.txt")

    return samples_path

def get_metrics_path(args, model_id):
    metrics_path = os.path.join(args.save_dir, "metrics", f"model_{model_id}.pkl")

    return metrics_path


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    return tokenizer


def get_model(args, model_id):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        revision=args.revision_template.format(revision_id=model_id+1)
    )

    return model


def get_sampling_prompts(args, seed=0, length=20, n_sample=1):
    dataset = load_dataset("roneneldan/TinyStories")
    tokenizer = get_tokenizer(args)

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


def write_metrics(metrics_path, metrics):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)


def read_metrics(metrics_path):
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    return metrics


def run_metrics(args, model, tokenizer, model_id):
    samples_path = get_samples_path(args)
    prompts, samples = read_samples(
        samples_path=samples_path,
    )

    # Evaluate each model revision
    metrics_path = get_metrics_path(args, model_id=model_id)
    with thing_exists_lock(path=metrics_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Metrics already exist, skipping evaluation")
        else:
            print(f"Evaluating model {model_id+1}/3")
            _, metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                texts=samples,
                metric=experiment_metric,
                prompts=prompts,
                batch_size=1,
            )
            write_metrics(metrics_path=metrics_path, metrics=metrics)


def run_testing(args):
    # Evaluate each model revision
    revision_metrics = []
    for model_id in range(3):
        metrics_path = get_metrics_path(args, model_id=model_id)
        metrics = read_metrics(metrics_path=metrics_path)
        revision_metrics.append(np.mean(metrics))

    stat = revision_metrics[args.sampling_model_id]
    mean = (sum(revision_metrics) - revision_metrics[args.sampling_model_id]) / 2
    std = abs(revision_metrics[(args.sampling_model_id + 1) % 3] - revision_metrics[(args.sampling_model_id - 1) % 3])

    z_score = (stat - mean) / (std + Z_SCORE_EPS)

    return z_score, revision_metrics


def run_sampling(args, model, tokenizer):
    samples_path = get_samples_path(args)
    with thing_exists_lock(path=samples_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Samples already exist, skipping sampling")
        else:
            # Configure sampling parameters
            sampling_params = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }

            if args.prompt is None:
                prompts = get_sampling_prompts(args, seed=args.sampling_seed, n_sample=args.n_sample)
            else:
                prompts = [args.prompt] * args.n_sample

            # Generate completions using model checkpoint
            print("Generating samples from model...")
            completions = generate(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
            )

            # Save generated texts
            write_samples(samples_path=samples_path, prompts=prompts, completions=completions)

            print(f"Saved {len(completions)} samples to {samples_path}")
    

def experiment_metric(tokenized_text, prediction, tokenized_prompt):
    if len(tokenized_text) <= len(tokenized_prompt):
        return 0.0
    return sum(prediction[len(tokenized_prompt):]).item()


def build_args(base_args, sweep_config):
    args = argparse.Namespace(**base_args.__dict__)
    for k, v in sweep_config.items():
        setattr(args, k, v)

    return args


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="1B")
    parser.add_argument("--sampling_model_id", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--prompt", type=str, default=None)

    base_args = parser.parse_args()

    assert base_args.model_size in MODEL_NAME_DICT.keys()
    base_args.model_name = MODEL_NAME_DICT[base_args.model_size]
    base_args.revision_template = REVISION_TEMPLATE_DICT[base_args.model_size]

    tokenizer = get_tokenizer(base_args)

    sweep_configs = {
        "n_sample": [1, 5, 10, 100, 200, 500, 1000],
        "sampling_seed": list(range(10)),
        "temperature": [1.0],
    }

    param_names = list(sweep_configs.keys())
    param_values = [sweep_configs[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    model = get_model(base_args, model_id=base_args.sampling_model_id)
    model.to('cuda'); model.eval()
    for _, params in enumerate(param_combinations):
        sweep_config = dict(zip(param_names, params))
        args = build_args(base_args, sweep_config)

        run_sampling(args, model, tokenizer)

    for model_id in range(3):
        model = get_model(base_args, model_id=model_id)
        model.to('cuda'); model.eval()
        for _, params in enumerate(param_combinations):
            sweep_config = dict(zip(param_names, params))
            args = build_args(base_args, sweep_config)
            run_metrics(args, model, tokenizer, model_id)

    for _, params in enumerate(param_combinations):
        sweep_config = dict(zip(param_names, params))
        args = build_args(base_args, sweep_config)

        z_score, _ = run_testing(args)

        experiment_log = {"z_score": z_score, "args": args}
        experiment_log_path = get_experiment_log_path(args)
        write_experiment_log(experiment_log_path=experiment_log_path, experiment_log=experiment_log)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()