import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import evaluate
import numpy as np
import scipy
from datasets import load_dataset
import pandas as pd

perplexity = evaluate.load("perplexity", module_type="metric")

def add_pplx(df, input_texts, model_id, model_nick, filename):
    print(f"Starting {model_nick} . . . ", end = ' ', flush=True)
    eval = perplexity.compute(model_id=model_id,
                                add_start_token=False,
                                predictions=input_texts)
    pplx = np.log(eval['perplexities'])

    df[model_nick] = pplx

    df = pd.DataFrame(df)

    df.to_csv(filename, index=False)
    print("done!",flush=True)

if __name__ == "__main__":
    
    model_id = "EleutherAI/pythia-6.9b"
    model_nick = "duped"
    filename = "pile_100k_duped_0102.csv"

    df = {}

    raw_dataset = load_dataset('EleutherAI/pile-deduped-pythia-random-sampled',split='train')

    input_tokens = list(raw_dataset['Tokens'])[:100000]
    order_ground = list(raw_dataset['Index'])[:100000]
    
    df['index'] = order_ground 

    tokenizer = AutoTokenizer.from_pretrained(
      model_id
    )
    input_texts = tokenizer.batch_decode(input_tokens)

    del raw_dataset
    del tokenizer

    add_pplx(df, input_texts, model_id, model_nick, filename)
