# Blackbox Model Provenance via Palimpsestic Membership Inference
This repository provides code for . 

Suppose **Alice** trains an open-weight language model, and subsequently **Bob** uses a blackbox derivative of
Alice’s model to produce text. Can Alice prove that
Bob is using her model, either by querying Bob’s derivative model (_query_ setting) or from the text alone (_observational_ setting)?

We propose tests using the **training order** of Alice's model and investigate it through the
lens of _palimpsestic memorization_ in language models: models are more likely
to memorize data seen later in training, so we can test whether Bob is using
Alice’s model using test statistics that capture correlation between the likelihood
of tokens in Bob’s text. 

Specifically, this repository provides code to: 
- compute p-values for the independence test given a model and a training dataset
- compute p-values for the independence test given some sampled text and a training dataset
- - with a partitioning method
  - and reshuffling method

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage 

We provide scripts for models we experiment on, Pythia and OLMo families, and for both settings. 

#### `scripts/query/pythia.py` 
This script runs the query setting test w/ a reference model (see Equation 2) using (a sample of) the Pythia training dataset. 
This script accepts these command-line arguments: 
- `--model`: HuggingFace model ID for the model to be audited.
- `--ref_model`: HuggingFace model ID for the reference model.
- `--n_samples`: Number of texts to use for the statistic.

#### `scripts/observation/partition.py` 
This script runs the observational setting test that partitions a model's training transcript based on data order. We provide a general script that accepts the path to 
an InfiniGram index (see https://infini-gram.readthedocs.io/en/latest/) and a list of texts (saved w/ pickle). 
This script accepts these command-line arguments: 
- `--texts_paths`: Path to the text samples to be audited. 
- `--infinigram_index_dir`: Path to local InfiniGram index. 
- `--n_texts`: Number of texts to use for the statistic.
- `--k`: Max. tokens for matching k-grams.
- `--tokenizer_name`: Tokenizer to tokenize the texts and used to build the index. 
