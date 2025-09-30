"""
Computes p-value from \phi_{query}^{ref} (see Equation 2).

Command-line arguments
  model (\mu_\beta): HuggingFace model ID
  ref_model (\mu_0): HuggingFace model ID
  n_samples (n): Number of sequences to sample.
  transcript: Name or path to a HuggingFace dataset that contains ordered
      training data samples, i.e., with an `index` and a `tokens` column.
      For example, the transcript could be any of the subset here
      https://huggingface.co/datasets/hij/sequence_samples, which are the
      transcripts used in our experiment.
  metric_column_name: If specified, use the precomputed metrics stored at
      the given column.
  ref_metric_column_name: If specified, use the precomputed metrics stored at
      the given column as references.

Example usage:
python blackbox-model-tracing/scripts/query/run_query_test.py \
    --model EleutherAI/pythia-6.9b-deduped \
    --ref_model EleutherAI/pythia-6.9b \
    --n_samples 100000 \
    --transcript hij/sequence_samples/pythia_deduped_100k

This tests whether EleutherAI/pythia-6.9b-deduped is trained on
the Pile deduped dataset.
It takes about 3 hrs to compute the logprob of the sequences on an A100.
The program should output a p-value around 1e-50.
"""
import sys
# Add the path to the blackbox-model-tracing dir.
sys.path.append('blackbox-model-tracing')


import argparse
import numpy as np

from datasets import load_dataset, load_from_disk
from tracing.index import DocumentIndex
from tracing.query.metrics import pplx 
from tracing.query.statistics import BasicStatistic
from transformers import AutoTokenizer


def load_transcript(transcript_name_or_path):
  try:
    # First try to load the transcript as a HF dataset.
    if transcript_name_or_path.count('/') < 2:
      dataset = load_dataset(transcript_name_or_path, split="train")
    else:
      # The dataset might contain subsets.
      dataset_name, subset_name = transcript_name_or_path.rsplit("/", 1)
      dataset = load_dataset(dataset_name, subset_name, split="train")
  except:
    # Try to load the transcript as a local dataset.
    dataset = load_from_disk(transcript_name_or_path) 
  return dataset


def phi_qr(args, document_index, metric_fn):
  phi = BasicStatistic(document_index, metric_fn, reference_path=args.ref_model)
  return phi(args.model)


if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str,
                      default="EleutherAI/pythia-6.9b-deduped")
  parser.add_argument("--ref_model", type=str,
                      default="EleutherAI/pythia-6.9b")
  parser.add_argument("--n_samples", type=int, default=100000)
  parser.add_argument("--transcript", type=str,
                      default="hij/sequence_samples/pythia_deduped_100k")
  parser.add_argument("--metric_column_name", type=str, default=None)
  parser.add_argument("--ref_metric_column_name", type=str, default=None) 
  args = parser.parse_args()

  transcript = load_transcript(args.transcript)

  tokens = list(transcript["tokens"])[:args.n_samples]
  order = list(transcript["index"])[:args.n_samples]

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  texts = tokenizer.batch_decode(tokens)

  document_index = DocumentIndex(texts, order)

  if not args.metric_column_name:
    # Compute metrics.
    print(phi_qr(args, document_index, pplx))
  else:
    # Use pre-computed metrics.
    metrics = list(transcript[args.metric_column_name])[:args.n_samples]
    metrics = np.array([np.mean(x) for x in metrics])
    ref_metrics = None
    if args.ref_metric_column_name:
      ref_metrics = list(
          transcript[args.ref_metric_column_name])[:args.n_samples]
      ref_metrics = np.array([np.mean(x) for x in ref_metrics])
    print(phi_qr(args, document_index,
                lambda x, y: metrics if x == args.model else ref_metrics))
