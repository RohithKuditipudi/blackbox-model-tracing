"""
tests I tried on the n-gram indices
"""
import numpy as np 
import pickle 
import scipy 
import matplotlib.pyplot as plt 
import pandas as pd 
import random

def get_ngram_mean_one(text):
    new_list = []
    for one_text_grams in text:
        for j in range(len(one_text_grams)):
            if len(one_text_grams[j]) != 1:
                continue 
            new_list.append(one_text_grams[j][0][0])
    return np.mean(new_list)

def ngram_permtest_one(text, T):
    unshuffled_mean = get_ngram_mean_one(text)
    print(unshuffled_mean)
    shuffled_means = []
    for t in range(T):
        shuffle_order = np.arange(100000)
        random.shuffle(shuffle_order)
        new_list = []
        
        for one_text_grams in text:
            for j in range(len(one_text_grams)):
                if len(one_text_grams[j]) != 1:
                    continue 
                new_list.append(shuffle_order[one_text_grams[j][0][0]])
        shuffled_means.append(np.mean(new_list))
        # print(np.mean(new_list))
    # print(unshuffled_mean)
    count = sum(1 for x in shuffled_means if x > unshuffled_mean) + 1
    return count / (T + 1)

"""
comparing the means using permutation test 
"""
def mean_perm_test(data, T=99):
  n_texts = [100, 500, 1000, 2000, 5000, 10000]
  for n in n_texts:
    print(n, end=' ')
    print(ngram_permtest_one(data[:n], T=999))

"""
using spearman test for p-values 
"""
def spearman_test(data):
  for n in n_texts:
    counts = np.zeros(100000)
    for text in data[:n]:
        for gram in text: 
            if len(gram) == 0: continue 
            # if len(gram[0]) != 1: continue
            for token in gram[0]: 
                counts[token] += 1 
    print(n, scipy.stats.spearmanr(np.arange(100000), counts))
  # plt.plot(np.arange(100000), counts)
