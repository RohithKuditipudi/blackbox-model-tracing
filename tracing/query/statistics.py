import numpy as np
import scipy

class TestStatistic:
    def __init__(self):
        pass

    def __call__(self,model_path,shuffle=False):
        pass
    
class BasicStatistic:
    def __init__(self,document_index,metric,n=None,reference_path=None):
        self.texts = document_index.index["texts"]
        self.order = document_index.index["order"]
        self.metric = metric
        self.n = n if n is not None else len(self.texts)
        self.reference_path = reference_path
    
    def __call__(self,model_path,shuffle=False):
        model_stats = eval_model(model_path,self.texts,self.metric)
        ref_stats = eval_model(self.reference_path,self.texts,self.metric)

        if shuffle:
            perm = np.random.permutation(self.n)
            order = perm[self.order]
        else:
            order = self.order

        return scipy.stats.spearmanr(np.argsort(order), model_stats-ref_stats)

def eval_model(model_path, texts, metric):
    stats = metric(model_path,texts)

    return stats