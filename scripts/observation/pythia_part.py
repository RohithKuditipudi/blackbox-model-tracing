"""
Prints p-value from \phi_{obs}^{part} (see Section 4.3.1) for Pythia models. 
Command-line argument with text_path (with pickle); infinigram_index_dir (path to indexed training data --- see tracing/index.py); 
n_texts = n. 

Ex:
python pythia_part.py --texts_paths gens.pkl --infinigram_index_dir /path/to/index --n_texts 100000

Uses a pre-built InfiniGram index for the Pythia model family, which efficiently maps n-grams to training documents occurences. 
(see https://infini-gram.readthedocs.io/en/latest/)
Computes the Spearman correlation for the number of counts of each n-gram in the training data versus train order (see Algorithm 2). 
We take \rho to be spearman rank correlation, \Chi to be n-gram count, and k to be the number of train steps (100000 for Pythia models). 
See tracing/observation/metrics.py for other metric options. 
"""

import argparse
import random 
import pickle 
from transformers import AutoTokenizer

from ..index import InfiniGramIndex
from ..tracing.observation.metrics import spearman_matches

def phi_op(index, texts, k):
  matched_steps = index.match_ngrams_to_steps_list(texts, n_max=k)
  n_train_steps = index.get_training_steps
  return spearman_matches(n_train_steps, matched_texts_to_steps)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--infinigram_index_dir', type=str, required=True)
  parser.add_argument('--tokenizer_name',
                      type=str,
                      default='EleutherAI/pythia-6.9b-deduped',
                      help='The name of the tokenizer used to build the index.')
  parser.add_argument('--k', type=int, default=8)
  parser.add_argument('--texts_path', type=str, required=True)
  parser.add_argument('--n_texts', type=int, default=5000)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  
  random.seed(args.seed)

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
  index = InfiniGramIndex(tokenizer)
  index.load_index(index_dir=args.infinigram_index_dir,
                   eos_token_id=tokenizer.eos_token_id)
  
  texts = pickle.load(open(args.texts_path, 'rb'))

  random.shuffle(texts)
  print(phi_op(index, texts[:args.n_texts], args.k))
  
