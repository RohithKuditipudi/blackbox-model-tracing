import numpy as np
import scipy 
from ..utils import flatten_list

def avg(matched_text_to_steps):
    return np.mean(flatten_list(matched_text_to_steps))

def spearman_matches(n_steps, matched_texts_to_steps):
    counts = np.zeros(n_steps)
    matched_text_to_steps = flatten_list(matched_texts_to_steps)
    for step in matched_text_to_steps:
        counts[step] += 1
        
    return scipy.stats.spearmanr(np.arange(n_steps), counts)

def stratified_avg(matched_text_to_steps):
    sampled_values = []
    for text in matched_text_to_steps:
        for pos in text:
            if pos:
                sampled_values.append(np.random.choice(matched_text_to_steps[text][pos]))
    
    return np.mean(sampled_values)

def single_match(matched_text_to_steps):
    single_values = []
    for text in matched_text_to_steps:
        for pos in text:
            if pos:
                if len(matched_text_to_steps[text][pos]) == 1:
                    single_values.append(matched_text_to_steps[text][pos][0])
    
    return np.mean(single_values)

