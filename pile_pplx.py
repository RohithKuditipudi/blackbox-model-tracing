"""
Computes the perplexity of N_SAMPLES from any dataset on HuggingFace given a list of models. (Here, uses the pile @ Line 58)
Models (Huggingface ID and a nickname to save them in a dataframe) @ Line 40
Can also change number of samples and tokenizer used (in this case everything is Pythia)
Perplexities will be saved @ FILENAME csv. Also saves the "index" which is the original training order of the samples. 
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import evaluate
import numpy as np
import scipy
from datasets import load_dataset
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# evaluate has a perplexity calculator
perplexity = evaluate.load("perplexity", module_type="metric")

N_SAMPLES = 100000
FILENAME = 'pile_pythia_pplx_100k.csv'

def add_pplx(input_texts, model_id, model_nick, filename):
    print(f"Starting {model_nick} . . . ", end = ' ', flush=True)

    df = pd.read_csv(filename)

    eval = perplexity.compute(model_id=model_id,
                                add_start_token=False,
                                predictions=input_texts)
    pplx = np.log(eval['perplexities'])

    df[model_nick] = pplx

    df.to_csv(filename, index=False)
    print("done!",flush=True)

if __name__ == "__main__":
    
    models = [
        ("EleutherAI/pythia-1.4b-deduped", "deduped-1-4b"),
        ("EleutherAI/pythia-2.8b-deduped", "deduped-2-8b"),
        ("EleutherAI/pythia-1b-deduped", "deduped-1b"),
        ("EleutherAI/pythia-410m-deduped", "deduped-410m"),
        ("EleutherAI/pythia-160m-deduped", "deduped-160m"),
        ("EleutherAI/pythia-70m-deduped", "deduped-70m"),
        ("EleutherAI/pythia-2.8b", "duped-2-8b"),
        ("EleutherAI/pythia-1.4b", "duped-1-4b"),
        ("EleutherAI/pythia-1b", "duped-1b"),
        ("EleutherAI/pythia-410m", "duped-410m"),
        ("EleutherAI/pythia-160m", "duped-160m"),
    ]

    df = pd.read_csv('pile_pythia_pplx_100k.csv')
    df = pd.DataFrame({})

    df.to_csv('pile_pythia_pplx_100k.csv', index=False)

    raw_dataset = load_dataset('EleutherAI/pile-deduped-pythia-random-sampled',split='train')

    input_tokens = list(raw_dataset['Tokens'])[:N_SAMPLES]
    order_ground = list(raw_dataset['Index'])[:N_SAMPLES]
    df['index'] = order_ground 

    tokenizer = AutoTokenizer.from_pretrained(
      "EleutherAI/pythia-6.9b-deduped"
    )
    input_texts = tokenizer.batch_decode(input_tokens)

    del raw_dataset
    del tokenizer

    for model in models:
        add_pplx(input_texts, model[0], model[1], FILENAME)
