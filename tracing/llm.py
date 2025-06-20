import torch
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
import random
from vllm import LLM, SamplingParams

def train_model(
    texts, 
    config, 
    tokenizer, 
    save_path, 
    index,
    batch_size=1, 
    epochs=1,
    reshuffle=False
):
    model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    random.seed(index)
    shuffle_orders = [random.shuffle(list(range(len(texts)))) for _ in range(epochs)]

    for epoch in range(epochs):
        if (epoch == 0) or reshuffle:
            shuffle_order = shuffle_orders[epoch] 
        shuffled_texts = [texts[i] for i in shuffle_order]

        train_dataloader = DataLoader(shuffled_texts, batch_size=batch_size, shuffle=False) # assume train_texts is shuffled in desired order
        batch_iterator = tqdm(train_dataloader)

        for batch_idx, batch in enumerate(batch_iterator): 
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inputs['labels'] = inputs['input_ids'].clone()

            outputs = model(**inputs)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "epoch": epoch,
            })

        model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))


def distill_model(
    teacher_path, 
    texts, 
    config, 
    tokenizer, 
    save_path, 
    index,
    batch_size=1, 
    epochs=1, 
    temperature=1.0, 
    hard_targets=False,
):
    """
    Perform knowledge distillation training, similar to train_tiny
    """

    student_model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    teacher_model = LlamaForCausalLM.from_pretrained(teacher_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    
    student_model.train()
    teacher_model.eval()

    random.seed(index)
    
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
        student_model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))


def generate(prompts, model_path, sampling_params, prompt_template="{prompt}"):
    """Generate synthetic text data using vLLM"""
    llm = LLM(model=model_path)
    
    prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]
    return generated_texts