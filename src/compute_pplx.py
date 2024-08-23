import gc
import json

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def get_pplx_at_checkpoints(sequences,
                            model_id,
                            revisions,
                            output_path_base,
                            cache_dir,
                            prefix_len=32,
                            window_len=64,
                            batch_size=32):
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
  tokenizer.pad_token = '<|padding|>'
  tokenizer.padding_side = 'left'
  torch_dtype = torch.bfloat16
  for revision in revisions:
    try:
      del model
    except NameError:
      pass
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 low_cpu_mem_usage=True,
                                                 device_map='auto',
                                                 cache_dir=cache_dir,
                                                 torch_dtype=torch_dtype,
                                                 revision=revision)
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
    torch.save(
        seq_to_pplx,
        f'{output_path_base}_p{prefix_len}_w{window_len}'
        f'_{model_id.split("/")[-1]}_{revision}.pt'
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
