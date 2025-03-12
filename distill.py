import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
import evaluate
import wandb

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import argparse
import json
import hashlib
import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def distill_tiny(teacher_model, texts, config, tokenizer, save_dir, df, index, df_path,
                     batch_size=1, epochs=1, temperature=1.0, hard_targets=False, eval_model=True):
    """
    Perform knowledge distillation training, similar to train_tiny
    """
    student_model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(epochs):
        train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)
        batch_iterator = tqdm(train_dataloader)
        
        for batch_idx, batch in enumerate(batch_iterator):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs).logits
                if hard_targets:
                    targets = torch.argmax(teacher_outputs, dim=-1)
                else:
                    targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)
            
            # Get student predictions
            student_outputs = student_model(**inputs).logits
            
            # Calculate distillation loss
            if hard_targets:
                loss = criterion(student_outputs.transpose(1,2),targets)
            else:
                loss = criterion(student_outputs.transpose(1, 2), targets.transpose(1, 2))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "epoch": epoch,
            })
        
        # Save checkpoint after each epoch
        student_model.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'))

        if eval_model:
            pplx = eval_tiny(os.path.join(save_dir, f'epoch-{epoch}-index-{index}'), texts)
            df[f'pplx-{index}-epoch-{epoch}'] = pplx

        df.to_csv(df_path)

def eval_tiny(model_path, eval_texts):
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=eval_texts)
    pplx = np.log(result['perplexities'])

    return pplx
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--teacher_path', type=str, required=True, help='Path to teacher model checkpoint')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--n_train_samples', type=int, default=20000, help='Number of training samples')
    parser.add_argument('--offset', type=int, default=10000, help='Offset')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--hard_targets', action='store_true', help='Use hard targets')
    parser.add_argument("--eval_model", action='store_true')
    
    args = parser.parse_args()

    args_dict = vars(args)
    args_dict['git_commit'] = get_git_revision_hash()
    args_str = json.dumps(args_dict, indent=2)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]

    N_TRAIN_SAMPLES = args.n_train_samples

    SAVE_DIR = args.save_dir
    REF_PATH = os.path.join(SAVE_DIR, f'tiny_dist_model_{args_hash}')
    DF_PATH = os.path.join(REF_PATH, f'tinystories.csv')

    os.makedirs(REF_PATH, exist_ok=True)
    
    with open(os.path.join(REF_PATH, 'args.json'), 'w') as f:
        f.write(args_str)
    
    wandb.init(
        project="blackbox-model-tracing",
        config=args_dict,
        name=f"distill_{args_hash}",
    )

    df = pd.DataFrame({})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Define model config
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )

    # Load teacher model
    teacher_model = LlamaForCausalLM.from_pretrained(args.teacher_path)
    
    # Prepare dataset
    dataset = load_dataset("roneneldan/TinyStories")
    texts = dataset["train"]["text"][args.offset:args.offset+N_TRAIN_SAMPLES]
    texts = [item for item in texts if item != ""]
    
    # Run distillation
    distill_tiny(
        teacher_model=teacher_model,
        texts=texts,
        config=config,
        tokenizer=tokenizer,
        save_dir=REF_PATH,
        df=df,
        index=0,
        df_path=DF_PATH,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature=args.temperature,
        hard_targets=args.hard_targets,
        eval_model=args.eval_model
    )
