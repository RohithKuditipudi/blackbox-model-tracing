from collections import defaultdict
import math
from tqdm import tqdm
import numpy as np
import torch

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, tokens):
        for i in tqdm(range(len(tokens) - self.n + 1)):
            ngram = tuple(tokens[i:i+self.n])
            self.ngram_counts[ngram] += 1
            self.context_counts[ngram[:-1]] += 1
            self.vocab.update(ngram)

    def probability(self, ngram):
        context = ngram[:-1]
        numerator = self.ngram_counts[ngram] + 1  # Add-one smoothing
        denominator = self.context_counts[context] + len(self.vocab)
        return numerator / denominator

def train_ngram_model(tokens, n):
    model = NGramModel(n)
    model.train(tokens)
    return model

def text_likelihood(model, tokenizer, text):
    tokens = text if isinstance(text, list) else tokenizer.encode(text)
    log_likelihood = 0
    for i in range(len(tokens) - model.n + 1):
        ngram = tuple(tokens[i:i+model.n])
        log_likelihood += math.log(model.probability(ngram))
    return log_likelihood

def text_perplexity(model, tokenizer, text):
    tokens = text if isinstance(text, list) else tokenizer.encode(text)
    likelihood = text_likelihood(model, tokenizer, text)
    return math.exp(-likelihood / (len(tokens) - model.n + 1))

def texts_likelihood(model, tokenizer, texts):
    likelihoods = []
    for text in tqdm(texts):
        if isinstance(text, str) and tokenizer:
            tokens = tokenizer.encode(text)
        elif isinstance(text, list):
            tokens = text
        else:
            raise ValueError("Input must be either a string (with tokenizer provided) or a list of tokens")
        
        log_likelihood = 0
        for i in range(len(tokens) - model.n + 1):
            ngram = tuple(tokens[i:i+model.n])
            log_likelihood += math.log(model.probability(ngram))
        likelihoods.append(log_likelihood)
    return likelihoods

# Similarly, let's create a function for perplexities
def texts_perplexity(model, tokenizer, texts):
    perplexities = []
    for text in texts:
        if isinstance(text, str) and tokenizer:
            tokens = tokenizer.encode(text)
        elif isinstance(text, list):
            tokens = text
        else:
            raise ValueError("Input must be either a string (with tokenizer provided) or a list of tokens")
        
        likelihood = text_likelihood(model, tokenizer, text)
        perplexity = math.exp(-likelihood / (len(tokens) - model.n + 1))
        perplexities.append(perplexity)
    return perplexities

def split_and_train(dataset, n, sequences_per_chunk=300000):
    models = {}
    
    for i in range(0, 100000000, 10000000):
        start = i
        end = i + 10000000
        
        range_data = dataset.filter(lambda x: start <= x['Index'] < end)
        range_data = range_data.select(range(min(len(range_data), sequences_per_chunk)))
        
        all_tokens = [token for sample in range_data['Tokens'] for token in sample]
        
        model = train_ngram_model(all_tokens, n)
        
        models[f"{start}-{end}"] = model
        
        print(f"Trained model for range {start}-{end} with {len(range_data)} sequences")
    
    return models

def get_split_and_train(dataset, start, end, n, sequences_per_chunk=300000):
    
    range_data = dataset.filter(lambda x: start <= x['Index'] < end)
    range_data = range_data.select(range(min(len(range_data), sequences_per_chunk)))
    
    all_tokens = [token for sample in range_data['Tokens'] for token in sample]
    # print(all_tokens[:100])
    # print(len(all_tokens))
    
    model = train_ngram_model(all_tokens, n)
        
    print(f"Trained model for range {start}-{end} with {len(range_data)} sequences")
    
    return model

def train_split(dataset, n):
    
    all_tokens = np.array(dataset).flatten()
    
    model = train_ngram_model(all_tokens, n)
        
    print(f"Trained model for full range on {len(dataset)} sequences, {len(all_tokens)} tokens")
    
    return model
