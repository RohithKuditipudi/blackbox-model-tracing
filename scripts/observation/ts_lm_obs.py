import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, LlamaConfig
import wandb
import random

from tracing.llm import train_model, generate, evaluate_model

def experiment_metric(text, prediction, prompt):
    return sum(prediction[len(prompt):])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4) # TODO: larger batch size
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=10)
    parser.add_argument("--include_prompt", action="store_true", default=False)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    texts = dataset["train"]["text"]

    # Shuffle the dataset
    random.seed(args.seed)
    random.shuffle(texts)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )

    # Create save directories
    base_save_path = os.path.join(args.save_dir, "base_model")
    os.makedirs(base_save_path, exist_ok=True)

    # Train base model on shuffled full dataset
    print("Training base model...")
    wandb.init(project="tinystories-training", name=f"base_model")
    base_model, _, _ = train_model(
        texts=texts,
        config=config,
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
    
    # Get first 100 texts from shuffled test set and truncate each to first 20 tokens
    prompts = [tokenizer.decode(tokenizer.encode(text)[:20]) for text in random.sample(dataset["test"]["text"], k=100)]
    
    # Configure sampling parameters
    sampling_params = {
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 0.9,
        "num_samples": 100,
    }

    # Generate completions using base model checkpoint
    base_model_path = os.path.join(base_save_path, f"epoch-{0}")
    generated_texts = generate(
        prompts=prompts,
        model_path=base_model_path,
        sampling_params=sampling_params
    )

    # Save generated texts
    samples_path = os.path.join(args.save_dir, "base_model_samples.txt")
    with open(samples_path, "w") as f:
        for prompt, completion in zip(prompts, generated_texts):
            f.write(f"PROMPT:\n{prompt}\n\nCOMPLETION:\n{completion}\n\n{'='*80}\n\n")
    
    print(f"Saved {len(generated_texts)} samples to {samples_path}")

    # Partition the shuffled dataset
    partition_size = len(texts) // args.num_partitions
    
    # Fine-tune on each partition
    for i in range(args.num_partitions):
        print(f"Fine-tuning on partition {i+1}/{args.num_partitions}")
        
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        partition_texts = texts[start_idx:end_idx]
        
        partition_save_path = os.path.join(args.save_dir, f"partition_{i}")
        os.makedirs(partition_save_path, exist_ok=True)

        if args.fine_tune:
            # Load the last checkpoint of base model for fine-tuning
            load_path = os.path.join(base_save_path, f"epoch-{0}")
        else:
            load_path = None
        
        wandb.init(project="tinystories-training", name=f"partition_{i}")
        train_model(
            texts=partition_texts,
            config=config,
            tokenizer=tokenizer,
            save_path=partition_save_path,
            index=args.seed + i, # different seed for each partition
            batch_size=args.batch_size,
            epochs=1,
            load_path=load_path,
            shuffle=False,
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
        partition_model_path = os.path.join(args.save_dir, f"partition_{i}", f"epoch-{0}")
        
        # Get predictions using evaluate_model
        predictions, metrics = evaluate_model(
            model_path=partition_model_path,
            texts=samples, # TODO: texts should include prompts
            metric=experiment_metric,
            prompts=prompts,
            batch_size=args.batch_size
        )

        partition_metrics.append(metrics)

if __name__ == "__main__":
    main()
