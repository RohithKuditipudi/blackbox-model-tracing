import getpass
import json
import numpy as np
import os
import random
import socket
import sys

import torch

lib_dir = 'src'
sys.path.append(lib_dir)
from compute_pplx import get_pplx_at_checkpoints


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
  USER = getpass.getuser()
  MACHINE = socket.gethostname().split('.')[0]
  FILE_SYS = 'nlp/scr'
  MODEL_DIR = f'/{FILE_SYS}/{USER}/models'
  print(f'Local dir={MODEL_DIR}')
  assert os.path.isdir(MODEL_DIR)
  # READ-ONLY. DO NOT WRITE TO THIS DIRECTORY.
  SCR_MODEL_DIR = '/nlp/scr/hij/memorization/models'

  set_seed(0)

  model_id = 'EleutherAI/pythia-6.9b-deduped'
  revisions = [
      'step5000',
      'step10000',
      'step20000',
      'step30000',
      'step40000',
      'step55000',
      'step70000',
      'step80000',
      'step95000',
      'step120000',
  ]
  seq_to_source = {}
  # Compute pplx for training examples occur at different steps.
  for step in [5, 25, 50, 75, 100]:
    data = json.load(
        open(
            os.path.join(
                SCR_MODEL_DIR,
                f'pythia-6.9b-deduped_pile_step{step}k_{step+1}k_sampled_200k'
                f'_prefix_len32_to_outputs_len32.json')))
    full_seqs = set([k + v['label'] for k, v in data.items()])
    for s in full_seqs:
      seq_to_source[s] = step

  print(f'Total #sequences={len(seq_to_source)}')
  get_pplx_at_checkpoints(list(seq_to_source),
                          model_id,
                          revisions,
                          os.path.join(MODEL_DIR, 'seq_to_pplx_1m'),
                          cache_dir=MODEL_DIR,
                          prefix_len=32,
                          window_len=64,
                          batch_size=32)
