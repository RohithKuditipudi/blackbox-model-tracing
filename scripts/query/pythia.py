"""
Prints p-value from \phi_{query}^{ref} (see Equation 2) for Pythia models. 
Command-line argument with model=\mu_\beta (HuggingFace ID); ref_model=\mu_0 (HuggingFace ID); n_samples = n

Uses HF dataset EleutherAI/pile-deduped-pythia-random-sampled, which contains 64-token documents frome The Pile and the 
corresponding train batch index (for each of the deduped family of models). 
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

from metrics import pplx 
from statistics import BasicStatistic
from ..index import DocumentIndex

def phi_qr(args, document_index):
  phi = BasicStatistic(document_index, pplx, reference_path=args.ref_model)
  return phi(args.model)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default='EleutherAI/pythia-6.9b-deduped')
  parser.add_argument("--ref_model", type=str, default='EleutherAI/pythia-6.9b')
  parser.add_argument("--n_samples", type=int, default=100000)
  args = parser.parse_args()

  pile_dataset = load_dataset('EleutherAI/pile-deduped-pythia-random-sampled',split='train')

  tokens = list(raw_dataset['Tokens'])[:args.n_samples]
  order = list(raw_dataset['Index'])[:args.n_samples]

  tokenizer = AutoTokenizer.from_pretrained(
      model_id
  )
  texts = tokenizer.batch_decode(tokens)

  document_index = DocumentIndex(texts, order)

  print(phi_qr(args, document_indx))
