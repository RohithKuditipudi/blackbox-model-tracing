# I ran: python this_script 34

from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer

import os
import argparse
import json
import hashlib
import subprocess
import pickle

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def index_tiny(texts, tokenizer, k, save_dir):
    index = defaultdict(list)
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        for i in range(len(tokens) - k + 1):
            kgram = tuple(tokens[i:i+k])
            index[kgram].append(i)
            
    index_file_path = os.path.join(save_dir, f'kgram_index_k{k}.pkl')
    
    with open(index_file_path, 'wb') as f:
        pickle.dump(index, f)
    
    print(f"Index saved to {index_file_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n_train_samples", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    args_dict = vars(args)
    args_dict['git_commit'] = get_git_revision_hash()
    args_str = json.dumps(args_dict, indent=2)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]

    N_TRAIN_SAMPLES = args.n_train_samples

    SAVE_DIR = args.save_dir
    INDEX_PATH = os.path.join(SAVE_DIR, f'kgram_index_{args_hash}')

    os.makedirs(INDEX_PATH, exist_ok=True)
    
    with open(os.path.join(INDEX_PATH, 'args.json'), 'w') as f:
        f.write(args_str)

    dataset = load_dataset("roneneldan/TinyStories")

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    texts = dataset["train"]["text"][:N_TRAIN_SAMPLES]
    texts = [item for item in texts if item != ""]

    index_tiny(
        texts, 
        tokenizer,
        args.k,
        INDEX_PATH,  
    )
