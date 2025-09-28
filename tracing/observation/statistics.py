import numpy as np

class TestStatistic:
    def __init__(self):
        pass

    def __call__(self,texts,shuffle=False):
        pass
    
class BasicNGramStatistic:
    def __init__(self,ngram_index,metric):
        self.index = ngram_index
        self.metric = metric
    
    def __call__(self,texts,shuffle=False):
        matched_text_to_steps = []
        for text in texts:
            matched_text_to_steps.append(self.index.match_ngrams_to_steps(text))

        if shuffle:
            perm = np.random.permutation(self.index.num_docs)
            for text in range(len(matched_text_to_steps)):
                for pos in range(len(matched_text_to_steps[text])):
                    matched_text_to_steps[text][pos] = [perm[doc_idx] for doc_idx in matched_text_to_steps[text][pos]]

        return self.metric(matched_text_to_steps)
