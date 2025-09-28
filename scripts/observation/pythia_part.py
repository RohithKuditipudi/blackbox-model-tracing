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
  
