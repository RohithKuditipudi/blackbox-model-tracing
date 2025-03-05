import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from datasets import load_dataset
import evaluate
import os
import pandas as pd

import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def distill_tiny(teacher_model, texts, config, tokenizer, save_dir, df, index,
                     batch_size=1, epochs=1, temperature=1.0):
    """
    Perform knowledge distillation training, similar to train_tiny
    """
    student_model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
    criterion = nn.KLDivLoss(reduction='batchmean')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(epochs):
        train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)
        batch_iterator = tqdm(train_dataloader)
        
        for batch in batch_iterator:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs).logits
                # print(teacher_outputs.shape)
                soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=-1)
            
            # Get student predictions
            student_outputs = student_model(**inputs).logits
            # print(student_outputs.shape)
            student_soft = nn.functional.log_softmax(student_outputs / temperature, dim=-1)
            
            # Calculate distillation loss
            loss = criterion(student_soft, soft_targets) * (temperature ** 2)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save checkpoint after each epoch
        student_model.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'epoch-{epoch}'))

        pplx = eval_tiny(os.path.join(save_dir, f'epoch-{epoch}'), texts)
        df[f'pplx-{index}-epoch-{epoch}'] = pplx

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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--n_train_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model')
    
    args = parser.parse_args()

    N_TRAIN_SAMPLES = args.n_train_samples

    SAVE_DIR = args.save_dir
    REF_PATH = os.path.join(SAVE_DIR, f'tiny_distilled_model')
    DF_PATH = os.path.join(REF_PATH, f'tinystories.csv')

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
    texts = dataset["train"]["text"][:N_TRAIN_SAMPLES]
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature=args.temperature
    )

    df.to_csv(DF_PATH)
