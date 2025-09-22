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
    assert args.num_teacher_checkpoints > 0, "num_teacher_checkpoints must be greater than 0"
    assert args.num_student_checkpoints > 0, "num_student_checkpoints must be greater than 0"
    assert args.num_distillation_checkpoints > 0, "num_distillation_checkpoints must be greater than 0"


def update_experiment_args(args):
    args.include_hash = str_to_bool(args.include_hash)
    args.git_hash = get_git_revision_hash() if args.include_hash else None

    args.teacher_model_path = get_teacher_checkpoint_path(args, args.teacher_checkpoint_idx)
    args.student_model_path = get_student_checkpoint_path(args, args.student_checkpoint_idx)
    args.reference_model_path = get_reference_checkpoint_path(args, args.student_checkpoint_idx)

    args.distillation_texts_path = get_distillation_texts_path(args)
    args.distillation_model_path = get_distillation_checkpoint_path(args, args.distillation_checkpoint_idx)

    args.use_reference_model = str_to_bool(args.use_reference_model)


def get_teacher_training_args(args):
    teacher_training_args = argparse.Namespace()

    teacher_training_args.save_dir = args.save_dir
    teacher_training_args.git_hash = args.git_hash
    teacher_training_args.seed = args.seed

    teacher_training_args.batch_size = args.batch_size
    teacher_training_args.learning_rate = args.learning_rate

    teacher_training_args.n_teacher = args.n_teacher
    teacher_training_args.num_teacher_checkpoints = args.num_teacher_checkpoints

    teacher_training_args.hidden_size = args.hidden_size
    teacher_training_args.intermediate_size = args.intermediate_size
    teacher_training_args.num_hidden_layers = args.num_hidden_layers
    teacher_training_args.num_attention_heads = args.num_attention_heads
    teacher_training_args.max_position_embeddings = args.max_position_embeddings

    return teacher_training_args


def get_student_training_args(args):
    student_training_args = argparse.Namespace()

    student_training_args.save_dir = args.save_dir
    student_training_args.git_hash = args.git_hash
    student_training_args.seed = args.seed
    
    student_training_args.batch_size = args.batch_size
    student_training_args.learning_rate = args.learning_rate

    student_training_args.n_teacher = args.n_teacher
    student_training_args.n_student = args.n_student
    student_training_args.num_student_checkpoints = args.num_student_checkpoints

    student_training_args.hidden_size = args.hidden_size
    student_training_args.intermediate_size = args.intermediate_size
    student_training_args.num_hidden_layers = args.num_hidden_layers
    student_training_args.num_attention_heads = args.num_attention_heads
    student_training_args.max_position_embeddings = args.max_position_embeddings

    return student_training_args


def get_reference_training_args(args):
    reference_training_args = argparse.Namespace()

    reference_training_args.save_dir = args.save_dir
    reference_training_args.git_hash = args.git_hash
    reference_training_args.seed = args.seed

    reference_training_args.batch_size = args.batch_size
    reference_training_args.learning_rate = args.learning_rate

    reference_training_args.n_teacher = args.n_teacher
    reference_training_args.n_student = args.n_student
    reference_training_args.num_student_checkpoints = args.num_student_checkpoints

    reference_training_args.hidden_size = args.hidden_size
    reference_training_args.intermediate_size = args.intermediate_size
    reference_training_args.num_hidden_layers = args.num_hidden_layers
    reference_training_args.num_attention_heads = args.num_attention_heads
    reference_training_args.max_position_embeddings = args.max_position_embeddings

    return reference_training_args


def get_teacher_sampling_args(args):
    teacher_sampling_args = argparse.Namespace()

    teacher_sampling_args.save_dir = args.save_dir
    teacher_sampling_args.git_hash = args.git_hash
    teacher_sampling_args.seed = args.seed

    teacher_sampling_args.n_teacher = args.n_teacher
    teacher_sampling_args.n_student = args.n_student
    teacher_sampling_args.n_distill = args.n_distill

    teacher_sampling_args.teacher_model_path = args.teacher_model_path
    
    teacher_sampling_args.temperature = args.temperature
    teacher_sampling_args.max_tokens = args.max_tokens
    teacher_sampling_args.prompt = args.prompt
    teacher_sampling_args.sampling_seed = args.sampling_seed

    return teacher_sampling_args


def get_distillation_args(args):
    distillation_args = argparse.Namespace()
    
    distillation_args.save_dir = args.save_dir
    distillation_args.git_hash = args.git_hash
    distillation_args.seed = args.seed

    distillation_args.batch_size = args.batch_size
    distillation_args.learning_rate = args.learning_rate

    distillation_args.student_model_path = args.student_model_path

    distillation_args.n_distill = args.n_distill
    distillation_args.num_distillation_checkpoints = args.num_distillation_checkpoints
    distillation_args.distillation_texts_path = args.distillation_texts_path

    return distillation_args


def get_testing_args(args):
    testing_args = argparse.Namespace()
    
    testing_args.save_dir = args.save_dir
    testing_args.git_hash = args.git_hash
    testing_args.seed = args.seed

    testing_args.batch_size = args.batch_size

    testing_args.n_teacher = args.n_teacher
    testing_args.n_test = args.n_test
    testing_args.use_reference_model = args.use_reference_model

    testing_args.distillation_model_path = args.distillation_model_path
    testing_args.reference_model_path = args.reference_model_path
    
    return testing_args


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_teacher_training_texts(args):
    dataset = get_dataset()

    # Pepare training texts
    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]

    # Shuffle training texts
    random.seed(args.seed)
    random.shuffle(texts)

    return texts[:args.n_teacher]


def get_student_training_texts(args):
    dataset = get_dataset()

    # Pepare training texts
    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]

    # Shuffle training texts
    random.seed(args.seed)
    random.shuffle(texts)

    return texts[args.n_teacher:args.n_teacher+args.n_student]


def get_distillation_texts(args):
    return read_texts(args.distillation_texts_path)


def get_sampling_prompts(args, length=20):
    dataset = get_dataset()

    # Pepare distillation texts
    texts = dataset["train"]["text"]
    texts = [item for item in texts if item != ""]

    # Shuffle training texts
    random.seed(args.seed)
    random.shuffle(texts)

    tokenizer = get_tokenizer()

    prompts = [
        tokenizer.decode(tokenizer.encode(text)[:length]) 
        for text in texts[args.n_teacher+args.n_student:args.n_teacher+args.n_student+args.n_distill]
    ]

    return prompts


def get_distillation_texts_path(args):
    teacher_sampling_args = get_teacher_sampling_args(args)

    teacher_sampling_hash = hash_args(teacher_sampling_args)
    distillation_texts_path = os.path.join(args.save_dir, "texts", teacher_sampling_hash, "distillation_texts.txt")

    return distillation_texts_path


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


def get_teacher_checkpoint_path(args, checkpoint_idx):
    teacher_training_args = get_teacher_training_args(args)

    teacher_training_hash = hash_args(teacher_training_args)
    teacher_checkpoint_path = os.path.join(
        args.save_dir, 
        "teacher_models", 
        teacher_training_hash, 
        f"checkpoint_{checkpoint_idx}"
    )

    return teacher_checkpoint_path

def get_student_checkpoint_path(args, checkpoint_idx):
    student_training_args = get_student_training_args(args)

    student_training_hash = hash_args(student_training_args)
    student_checkpoint_path = os.path.join(
        args.save_dir, 
        "student_models",
        student_training_hash, 
        f"checkpoint_{checkpoint_idx}"
    )

    return student_checkpoint_path


def get_reference_checkpoint_path(args, checkpoint_idx):
    reference_training_args = get_reference_training_args(args)

    reference_training_hash = hash_args(reference_training_args)
    reference_checkpoint_path = os.path.join(
        args.save_dir, 
        "reference_models", 
        reference_training_hash, 
        f"checkpoint_{checkpoint_idx}"
    )

    return reference_checkpoint_path


def get_distillation_checkpoint_path(args, checkpoint_idx):
    distillation_args = get_distillation_args(args)

    distillation_hash = hash_args(distillation_args)
    distillation_checkpoint_path = os.path.join(
        args.save_dir, 
        "distillation_models", 
        distillation_hash, 
        f"checkpoint_{checkpoint_idx}"
    )

    return distillation_checkpoint_path


def get_metrics_path(args):
    testing_args = get_testing_args(args)

    testing_hash = hash_args(testing_args)
    metrics_path = os.path.join(args.save_dir, "metrics", testing_hash, "metrics.pkl")

    return metrics_path


def get_reference_metrics_path(args):
    testing_args = get_testing_args(args)

    testing_hash = hash_args(testing_args)
    metrics_path = os.path.join(args.save_dir, "metrics", testing_hash, "reference_metrics.pkl")

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


def run_teacher_training(args):
    tokenizer = get_tokenizer()
    texts = get_teacher_training_texts(args)

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

    final_teacher_checkpoint_path = get_teacher_checkpoint_path(
        args, 
        checkpoint_idx=args.num_teacher_checkpoints-1
    )
    with thing_exists_lock(
        path=final_teacher_checkpoint_path, 
        thing_exists_fn=model_exists
    ) as thing_exists:
        if thing_exists:
            print("Teacher model already exists, skipping training")
        else:
            # Train base model on shuffled full dataset
            print("Training teacher model...")
            wandb.init(project="tinystories-distillation", name=f"teacher_model")
            model, optimizer = None, None
            for i in range(args.num_teacher_checkpoints):
                teacher_checkpoint_path = get_teacher_checkpoint_path(args, checkpoint_idx=i)
                n_checkpoint, interval_size = get_n_checkpoint(
                    n_train=args.n_teacher, 
                    num_checkpoints=args.num_teacher_checkpoints, 
                    checkpoint_idx=i
                )

                train_model(
                    texts=texts[:n_checkpoint][-interval_size:],
                    config=config,
                    tokenizer=tokenizer,
                    save_path=teacher_checkpoint_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                    optimizer_params=optimizer_params,
                    model=model,
                    optimizer=optimizer,
                )
                model, optimizer = load_model_and_optimizer(teacher_checkpoint_path)
            wandb.finish()


def run_student_training(args):
    tokenizer = get_tokenizer()
    texts = get_student_training_texts(args)

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1e-6,
    )
    optimizer_params = get_default_optimizer_params(args)

    final_student_checkpoint_path = get_student_checkpoint_path(
        args, 
        checkpoint_idx=args.num_student_checkpoints-1
    )
    with thing_exists_lock(
        path=final_student_checkpoint_path, 
        thing_exists_fn=model_exists
    ) as thing_exists:
        if thing_exists:
            print("Student model already exists, skipping training")
        else:
            print("Training student model...")
            wandb.init(project="tinystories-distillation", name=f"student_model")
            model, optimizer = None, None
            for i in range(args.num_student_checkpoints):
                student_checkpoint_path = get_student_checkpoint_path(args, checkpoint_idx=i)
                n_checkpoint, interval_size = get_n_checkpoint(
                    n_train=args.n_student, 
                    num_checkpoints=args.num_student_checkpoints, 
                    checkpoint_idx=i
                )
                train_model(
                    texts=texts[:n_checkpoint][-interval_size:],
                    config=config,
                    tokenizer=tokenizer,
                    save_path=student_checkpoint_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                    optimizer_params=optimizer_params,
                    model=model,
                    optimizer=optimizer,
                )
                model, optimizer = load_model_and_optimizer(student_checkpoint_path)
            wandb.finish()


def run_reference_training(args):
    tokenizer = get_tokenizer()
    texts = get_student_training_texts(args)

    random.seed(args.seed + 12345)
    random.shuffle(texts) # train reference model on shuffled student training texts

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1e-6,
    )
    optimizer_params = get_default_optimizer_params(args)

    final_reference_checkpoint_path = get_reference_checkpoint_path(
        args, 
        checkpoint_idx=args.num_student_checkpoints-1
    )
    with thing_exists_lock(
        path=final_reference_checkpoint_path, 
        thing_exists_fn=model_exists
    ) as thing_exists:
        if thing_exists:
            print("Reference model already exists, skipping training")
        else:
            print("Training reference model...")
            wandb.init(project="tinystories-distillation", name=f"reference_model")
            model, optimizer = None, None
            for i in range(args.num_student_checkpoints):
                reference_checkpoint_path = get_reference_checkpoint_path(args, checkpoint_idx=i)
                n_checkpoint, interval_size = get_n_checkpoint(
                    n_train=args.n_student, 
                    num_checkpoints=args.num_student_checkpoints, 
                    checkpoint_idx=i
                )
                train_model(
                    texts=texts[:n_checkpoint][-interval_size:],
                    config=config,
                    tokenizer=tokenizer,
                    save_path=reference_checkpoint_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                    optimizer_params=optimizer_params,
                    model=model,
                    optimizer=optimizer,
                )
                model, optimizer = load_model_and_optimizer(reference_checkpoint_path)
            wandb.finish()


def run_teacher_sampling(args):
    # Configure sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    prompts = get_sampling_prompts(args)

    distillation_texts_path = get_distillation_texts_path(args)
    with thing_exists_lock(path=distillation_texts_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Teacher samples already exist, skipping sampling")
        else:
            # Generate completions using model checkpoint
            print("Generating samples from model...")
            completions = generate(
                prompts=prompts,
                model_checkpoint_path=os.path.join(args.teacher_model_path, "epoch-0"),
                sampling_params=SamplingParams(**sampling_params),
                seed=args.sampling_seed,
            )
            distillation_texts = [prompt + completion for prompt, completion in zip(prompts, completions)]

            # Save generated texts
            write_texts(texts_path=distillation_texts_path, texts=distillation_texts)
            print(f"Saved {len(distillation_texts)} texts to {distillation_texts_path}")


def run_distillation(args):
    # Load distillation texts to start distillation
    tokenizer = get_tokenizer()
    texts = get_distillation_texts(args)

    final_distillation_checkpoint_path = get_distillation_checkpoint_path(
        args, 
        checkpoint_idx=args.num_distillation_checkpoints-1
    )
    with thing_exists_lock(
        path=final_distillation_checkpoint_path, 
        thing_exists_fn=model_exists
    ) as thing_exists:
        if thing_exists:
            print("Distillation model already exists, skipping distillation")
        else:
            print("Distilling teacher model into student model...")
            wandb.init(project="tinystories-distillation", name=f"distillation_model")
            model, optimizer = load_model_and_optimizer(args.student_model_path)
            for i in range(args.num_distillation_checkpoints):
                distillation_checkpoint_path = get_distillation_checkpoint_path(args, checkpoint_idx=i)
                n_checkpoint, interval_size = get_n_checkpoint(
                    n_train=args.n_distill,
                    num_checkpoints=args.num_distillation_checkpoints,
                    checkpoint_idx=i
                )
                train_model(
                    texts=texts[:n_checkpoint][-interval_size:],
                    tokenizer=tokenizer,
                    save_path=distillation_checkpoint_path,
                    batch_size=args.batch_size,
                    epochs=1,
                    shuffle=False,
                    model=model,
                    optimizer=optimizer,
                )
                model, optimizer = load_model_and_optimizer(distillation_checkpoint_path)
            wandb.finish()


def run_testing(args):
    tokenizer = get_tokenizer()
    teacher_training_texts = get_teacher_training_texts(args)

    metrics_path = get_metrics_path(args)
    with thing_exists_lock(path=metrics_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists:
            print("Shuffle metrics already exists, skipping to p-value calculation")
        else:
            distillation_model, _ = load_model_and_optimizer(args.distillation_model_path)
            _, metrics = evaluate_model(
                model=distillation_model,
                tokenizer=tokenizer,
                texts=teacher_training_texts,
                metric=experiment_metric,
                batch_size=args.batch_size
            )
            write_metrics(metrics_path=metrics_path, metrics=metrics)
    
    reference_metrics_path = get_reference_metrics_path(args)
    with thing_exists_lock(path=reference_metrics_path, thing_exists_fn=file_exists) as thing_exists:
        if thing_exists or not args.use_reference_model:
            print("Reference metrics already exists (or not used), skipping to p-value calculation")
        else:
            reference_model, _ = load_model_and_optimizer(args.reference_model_path)
            _, reference_metrics = evaluate_model(
                model=reference_model,
                tokenizer=tokenizer,
                texts=teacher_training_texts,
                metric=experiment_metric,
                batch_size=args.batch_size
            )
            write_metrics(metrics_path=reference_metrics_path, metrics=reference_metrics)
        
    metrics = read_metrics(metrics_path=metrics_path)
    subsampled_indices = random.sample(range(len(teacher_training_texts)), args.n_test)

    if args.use_reference_model:
        reference_metrics = read_metrics(metrics_path=reference_metrics_path)
        subsampled_metrics = [metrics[i] - reference_metrics[i] for i in subsampled_indices]
    else:
        subsampled_metrics = [metrics[i] for i in subsampled_indices]

    _, p_value = scp.stats.spearmanr(subsampled_metrics, subsampled_indices)

    return p_value


def experiment_metric(tokenized_text, prediction, tokenized_prompt):
    if len(tokenized_text) <= len(tokenized_prompt):
        return 0.0
    return sum(prediction[len(tokenized_prompt):]).item()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Knowledge Distillation')

    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_teacher', type=int, default=8, help='Number of training samples')
    parser.add_argument('--n_student', type=int, default=8, help='Number of student samples')
    parser.add_argument('--n_distill', type=int, default=8, help='Number of distillation samples')
    parser.add_argument('--num_teacher_checkpoints', type=int, default=1, help='Number of teacher checkpoints')
    parser.add_argument('--num_student_checkpoints', type=int, default=1, help='Number of student checkpoints')
    parser.add_argument('--num_distillation_checkpoints', type=int, default=1, help='Number of distillation checkpoints')
    parser.add_argument('--teacher_checkpoint_idx', type=int, default=0, help='Teacher checkpoint index')
    parser.add_argument('--student_checkpoint_idx', type=int, default=0, help='Student checkpoint index')
    parser.add_argument('--distillation_checkpoint_idx', type=int, default=0, help='Distillation checkpoint index')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--sampling_seed', type=int, default=0, help='Sampling seed')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt')
    parser.add_argument('--max_tokens', type=int, default=64, help='Maximum tokens')
    parser.add_argument('--n_test', type=int, default=1, help='Number of test samples')
    parser.add_argument('--use_reference_model', type=str, default="true", help='Use reference model')
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

    teacher_training_args = get_teacher_training_args(args)
    run_teacher_training(teacher_training_args)

    student_training_args = get_student_training_args(args)
    run_student_training(student_training_args)

    reference_training_args = get_reference_training_args(args)
    run_reference_training(reference_training_args)

    teacher_sampling_args = get_teacher_sampling_args(args)
    run_teacher_sampling(teacher_sampling_args)

    distillation_args = get_distillation_args(args)
    run_distillation(distillation_args)

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