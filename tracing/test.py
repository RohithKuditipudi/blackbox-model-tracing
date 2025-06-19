import numpy as np
import scipy

def exact_permutation_test(T, statistic, **statistic_kwargs):
    original_statistic = statistic(**statistic_kwargs)
    rank = 0
    for _ in range(T):
        exchangeable_statistic = statistic(**statistic_kwargs,shuffle=True)
        if exchangeable_statistic >= original_statistic:
            rank += 1
    
    return (rank+1)/(T+1)

def approximate_permutation_test(T, statistic, **statistic_kwargs):
    original_statistic = statistic(**statistic_kwargs)
    mean = 0

    exchangeable_statistics = []
    for _ in range(T):
        exchangeable_statistics.append(statistic(**statistic_kwargs,shuffle=True))
    
    z_score = (original_statistic - np.mean(exchangeable_statistics)) / np.std(exchangeable_statistics)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))
    
    return p_value