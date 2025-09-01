import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams
import random
import numpy as np

from tracing.llm import generate, evaluate_model

import torch.distributed as dist

Z_SCORE_EPS = 1e-6

MODEL_NAME_DICT = {
    "1B": "allenai/OLMo-2-0425-1B",
}
REVISION_TEMPLATE_DICT = {
    "1B": "stage2-ingredient{revision_id}-step23852-tokens50B",
}

def get_samples_path(args):
    samples_path = os.path.join(args.save_dir, "samples.txt")

    return samples_path


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    return tokenizer


def get_model(args, revision_id):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.revision_template.format(revision_id=revision_id))

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


def run_testing(args):
    tokenizer = get_tokenizer(args)
    samples_path = get_samples_path(args)
    prompts, samples = read_samples(
        samples_path=samples_path,
    )

    # Evaluate each model revision
    revision_metrics = []
    for i in range(3):
        print(f"Evaluating model revision {i+1}/3")
        tmp_model = get_model(args, revision_id=i+1)
            
        _, metrics = evaluate_model(
            model=tmp_model,
            tokenizer=tokenizer,
            texts=samples,
            metric=experiment_metric,
            prompts=prompts,
            batch_size=1,
        )

        revision_metrics.append(np.mean(metrics))

    stat = revision_metrics[args.sampling_revision_id]
    mean = (sum(revision_metrics) - revision_metrics[args.sampling_revision_id]) / 2
    std = abs(revision_metrics[(args.sampling_revision_id + 1) % 3] - revision_metrics[(args.sampling_revision_id - 1) % 3])

    z_score = (stat - mean) / (std + Z_SCORE_EPS)

    return z_score, revision_metrics


def run_sampling(args):
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
        model_checkpoint_path=args.model_name,
        sampling_params=SamplingParams(**sampling_params),
        seed=args.sampling_seed,
        revision=args.revision_template.format(revision_id=args.sampling_revision_id),
    )

    # Save generated texts
    samples_path = get_samples_path(args)
    write_samples(samples_path=samples_path, prompts=prompts, completions=completions)

    print(f"Saved {len(completions)} samples to {samples_path}")
    

def experiment_metric(tokenized_text, prediction, tokenized_prompt):
    if len(tokenized_text) <= len(tokenized_prompt):
        return 0.0
    return sum(prediction[len(tokenized_prompt):]).item()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="1B")
    parser.add_argument("--sampling_revision_id", type=int, default=1)
    parser.add_argument("--sampling_seed", type=int, default=0)
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--num_shuffles", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)

    args = parser.parse_args()

    assert args.model in MODEL_NAME_DICT.keys()
    args.model_name = MODEL_NAME_DICT[args.model]
    args.revision_template = REVISION_TEMPLATE_DICT[args.model]

    run_sampling(args)
    z_score, _ = run_testing(args)

    print(f"z-score: {z_score}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()