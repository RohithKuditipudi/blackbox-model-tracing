"""
Prints p-value from \phi_{query}^{ref} (see Equation 2). 
Command-line argument with
  model (\mu_\beta): HuggingFace model ID
  ref_model (\mu_0): HuggingFace model ID
  n_samples (n): Number of sequences to sample.
  dataset_name: The name of the pre-training dataset.
      Must be one of the subset in the following dataset:
      https://huggingface.co/datasets/hij/sequence_samples.

Example:
python blackbox-model-tracing/scripts/query/run_query_test.py \
    --model EleutherAI/pythia-6.9b-deduped \
    --ref_model EleutherAI/pythia-6.9b \
    --n_samples 100000 \
    --dataset_name pythia_deduped_100k

It takes about 3 hrs to compute the logprob of the sequences on an A100.
The program should output a p-value around 1e-50.
"""
import sys
# Add the path to the blackbox-model-tracing dir.
sys.path.append('blackbox-model-tracing')


import argparse

from datasets import load_dataset
from tracing.index import DocumentIndex
from tracing.query.metrics import pplx 
from tracing.query.statistics import BasicStatistic
from transformers import AutoTokenizer


def phi_qr(args, document_index):
  phi = BasicStatistic(document_index, pplx, reference_path=args.ref_model)
  return phi(args.model)


if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="EleutherAI/pythia-6.9b-deduped")
  parser.add_argument("--ref_model", type=str, default="EleutherAI/pythia-6.9b")
  parser.add_argument("--n_samples", type=int, default=100000)
  parser.add_argument("--dataset_name", type=str, default="pythia_deduped_100k")
  args = parser.parse_args()

  hf_sequence_samples = "hij/sequence_samples"
  dataset = load_dataset(hf_sequence_samples, args.dataset_name, split="train")

  tokens = list(dataset["tokens"])[:args.n_samples]
  order = list(dataset["index"])[:args.n_samples]

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  texts = tokenizer.batch_decode(tokens)

  document_index = DocumentIndex(texts, order)

  print(phi_qr(args, document_index))
