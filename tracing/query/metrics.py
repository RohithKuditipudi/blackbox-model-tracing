import numpy as np
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def eval_model(model_path, texts, metric):
    stats = metric(model_path,texts)

    return stats

def pplx(model_path, texts):
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=texts)
    pplx = np.log(result['perplexities'])

    return pplx

def compute_per_token_pplx(model, encoded_inputs, labels):
   with torch.no_grad():
      outputs = model(encoded_inputs['input_ids'], labels=labels)
      loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
      shift_logits = outputs.logits[:, :-1, :].contiguous()
      labels = labels[:, 1:].contiguous()
      loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                    labels.view(-1))
      loss = loss.view(labels.size(0), -1)
      return loss

def get_pplx(sequences,
             model_id,
             revision='main',
             prefix_len=32,
             window_size=32,
             batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = '<|padding|>'
    tokenizer.padding_side = 'left'
    torch_dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map='auto',
        torch_dtype=torch_dtype,
        revision=revision
    )
    model = model.eval()
    
    seq_to_pplx = {}
    for i in tqdm(range(0, len(sequences), batch_size)):
        encoded_inputs = tokenizer(sequences[i:i + batch_size],
                                    return_tensors='pt',
                                    max_length=96,
                                    truncation=True,
                                    padding='max_length').to(model.device)
        labels = encoded_inputs['input_ids'].clone()
        labels[:, :prefix_len] = -100
        pplx = compute_per_token_pplx(model, encoded_inputs, labels)
        for b_i in range(len(pplx)):
            seq_to_pplx[sequences[i + b_i]] = pplx[b_i]

    pplx = []
    for seq in seq_to_pplx.keys():
        pplx.append(seq_to_pplx[seq][prefix_len : prefix_len + window_size].mean().tolist())

    return pplx.mean()
