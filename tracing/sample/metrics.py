import numpy as np
from ..utils import flatten_list

def avg(matched_text_to_steps):
    return np.mean(flatten_list(matched_text_to_steps))

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
