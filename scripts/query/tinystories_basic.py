import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
import wandb
import random
import pickle
import hashlib
import scipy as scp

from vllm import SamplingParams

from tracing.llm import train_model, evaluate_model, load_model_and_optimizer, model_exists, generate
from tracing.utils import get_git_revision_hash, thing_exists_lock, file_exists, str_to_bool

import torch.distributed as dist

def hash_args(args, length=16):
    return hashlib.sha256(str(args).encode()).hexdigest()[:length]


def validate_experiment_args(args):
    assert args.num_checkpoints > 0, "num_checkpoints must be greater than 0"


def update_experiment_args(args):
    args.include_hash = str_to_bool(args.include_hash)
    args.git_hash = get_git_revision_hash() if args.include_hash else None

    args.model_path = get_checkpoint_path(args, args.checkpoint_idx)


def get_training_args(args):
    training_args = argparse.Namespace()

    training_args.save_dir = args.save_dir
    training_args.git_hash = args.git_hash
    training_args.seed = args.seed

    training_args.batch_size = args.batch_size
    training_args.learning_rate = args.learning_rate

    training_args.n_train = args.n_train
    training_args.num_checkpoints = args.num_checkpoints

    training_args.hidden_size = args.hidden_size
    training_args.intermediate_size = args.intermediate_size
    training_args.num_hidden_layers = args.num_hidden_layers
    training_args.num_attention_heads = args.num_attention_heads
    training_args.max_position_embeddings = args.max_position_embeddings

    return training_args


def get_testing_args(args):
    testing_args = argparse.Namespace()
    
    testing_args.save_dir = args.save_dir
    testing_args.git_hash = args.git_hash
    testing_args.seed = args.seed

    testing_args.batch_size = args.batch_size

    testing_args.n_train = args.n_train
    testing_args.n_test = args.n_test

    testing_args.model_path = args.model_path
    
    return testing_args


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_training_texts(args):
    dataset = get_dataset()

    # Pepare training texts
    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]

    # Shuffle training texts
    random.seed(args.seed)
    random.shuffle(texts)

    return texts[:args.n_train]


def get_dataset():
    dataset = load_dataset("roneneldan/TinyStories")

    return dataset


def get_n_checkpoint(n_train, num_checkpoints, checkpoint_idx):
    interval_size = get_interval_size(n_train, num_checkpoints)
    n_checkpoint = (checkpoint_idx+1) * interval_size

    return n_checkpoint, interval_size


def get_interval_size(n_train, num_checkpoints):
    assert (n_train) % (num_checkpoints) == 0, "num_checkpoints must be a factor of n_train"
    interval_size = n_train // num_checkpoints

    return interval_size


def get_default_optimizer_params(args):
    return {"lr": args.learning_rate}


def get_checkpoint_path(args, checkpoint_idx):
    training_args = get_training_args(args)

    training_hash = hash_args(training_args)
    checkpoint_path = os.path.join(
        args.save_dir, 
        "models", 
        training_hash, 
        f"checkpoint_{checkpoint_idx}"
    )

    return checkpoint_path


def get_metrics_path(args):
    testing_args = get_testing_args(args)

    testing_hash = hash_args(testing_args)
    metrics_path = os.path.join(args.save_dir, "metrics", testing_hash, "metrics.pkl")

    return metrics_path


def get_experiment_log_path(args):
    experiment_hash = hash_args(args)
    experiment_log_path = os.path.join(args.save_dir, "experiment_logs", experiment_hash, "log.pkl")

    return experiment_log_path


def write_texts(texts_path, texts):
    os.makedirs(os.path.dirname(texts_path), exist_ok=True)
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)


def read_texts(texts_path):
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    return texts


def write_metrics(metrics_path, metrics):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)


def read_metrics(metrics_path):
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    return metrics


def write_experiment_log(experiment_log_path, experiment_log):
    os.makedirs(os.path.dirname(experiment_log_path), exist_ok=True)
    with open(experiment_log_path, "wb") as f:
        pickle.dump(experiment_log, f)


def run_training(args):
    tokenizer = get_tokenizer()
    texts = get_training_texts(args)

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

    final_teacher_checkpoint_path = get_checkpoint_path(
        args, 
        checkpoint_idx=args.num_teacher_checkpoints-1
    )
    with thing_exists_lock(
        path=final_teacher_checkpoint_path, 
        thing_exists_fn=model_exists
    ) as thing_exists:
        if thing_exists:
            print("Model already exists, skipping training")
        else:
            # Train base model on shuffled full dataset
            print("Training model...")
            wandb.init(project="tinystories-basic", name=f"model")
            model, optimizer = None, None
            for i in range(args.num_teacher_checkpoints):
                checkpoint_path = get_checkpoint_path(args, checkpoint_idx=i)
                n_checkpoint, interval_size = get_n_checkpoint(
                    n_train=args.n_train, 
                    num_checkpoints=args.num_checkpoints, 
                    checkpoint_idx=i
                )

                train_model(
                    texts=texts[:n_checkpoint][-interval_size:],
                    config=config,
                    tokenizer=tokenizer,
                    save_path=checkpoint_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                    optimizer_params=optimizer_params,
                    model=model,
                    optimizer=optimizer,
                )
                model, optimizer = load_model_and_optimizer(checkpoint_path)
            wandb.finish()


def run_testing(args):
    tokenizer = get_tokenizer()
    texts = get_training_texts(args)

    metrics_path = get_metrics_path(args)
    with thing_exists_lock(path=metrics_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Shuffle metrics already exists, skipping to z-score calculation")
        else:
            model, _ = load_model_and_optimizer(args.model_path)
            _, metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                metric=experiment_metric,
                batch_size=args.batch_size
            )
            write_metrics(metrics_path=metrics_path, metrics=metrics)
        
    metrics = read_metrics(metrics_path=metrics_path)

    subsampled_indices = random.sample(range(len(texts)), args.n_test)
    subsampled_metrics = [metrics[i] for i in subsampled_indices]

    _, p_value = scp.stats.spearmanr(subsampled_metrics, subsampled_indices)

    return p_value


def experiment_metric(tokenized_text, prediction, tokenized_prompt):
    if len(tokenized_text) <= len(tokenized_prompt):
        return 0.0
    return sum(prediction[len(tokenized_prompt):]).item()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic query setting')

    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_train', type=int, default=8, help='Number of training samples')
    parser.add_argument('--num_checkpoints', type=int, default=1, help='Number of teacher checkpoints')
    parser.add_argument('--checkpoint_idx', type=int, default=0, help='Teacher checkpoint index')
    parser.add_argument('--n_test', type=int, default=1, help='Number of test samples')
    parser.add_argument('--reference_model', type=str, default="false", help='Use reference model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--intermediate_size', type=int, default=512, help='Intermediate size')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--max_position_embeddings', type=int, default=512, help='Maximum position embeddings')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument("--include_hash", type=str, default="false")

    args = parser.parse_args()

    update_experiment_args(args)
    validate_experiment_args(args)

    training_args = get_training_args(args)
    run_training(training_args)

    testing_args = get_testing_args(args)
    p_value = run_testing(testing_args)

    print(f"p-value: {p_value}")

    experiment_log = {"p_value": p_value, "args": args}
    experiment_log_path = get_experiment_log_path(args)
    with thing_exists_lock(
        path=experiment_log_path, 
        thing_exists_fn=file_exists
    ) as thing_exists:
        if not thing_exists:
            write_experiment_log(experiment_log_path=experiment_log_path, experiment_log=experiment_log)

    if dist.is_initialized():
        dist.destroy_process_group()