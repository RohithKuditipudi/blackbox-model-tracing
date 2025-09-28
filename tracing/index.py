# change to transcript.py?

from collections import defaultdict

import json 

from infini_gram.engine import InfiniGramEngine
import timeit

from .utils import timeout

class DocumentIndex:
    def __init__(self, texts, order):
        self.num_docs = len(texts)
        self.index = {
            "texts": texts,
            "order": order,
        }
    
    def get_training_steps(self, input_ids):
        pass

class NGramIndex:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.num_docs = None

    def get_training_steps(self, input_ids):
        pass

    def match_ngrams_to_steps(self, text, n_max=None, print_stats=False):
        tokens = self.tokenizer.encode(text)
        if n_max is None:
            n_max = len(tokens)

        matched_steps = []
        for pos in range(len(tokens)-n_max):
            n = n_max
            while n > 0:
                ngram = tokens[pos:pos+n+1]
                start_time = timeit.default_timer()
                steps = self.get_training_steps(ngram)
                end_time = timeit.default_timer()
                if print_stats:
                    print("INDEX STATS:")
                    print(f"Looked for the following {n}-gram: {repr(self.tokenizer.decode(ngram))}")
                    print(f"Found {len(steps)} matches in {end_time - start_time} seconds")

                if len(steps) > 0:
                    matched_steps.append(steps)
                    break
                n -= 1
                        
        return matched_steps

    def match_ngrams_to_steps_list(self, texts, n_max=None, print_stats=False):
        full_matches = []
        
        for i in range(len(texts)):
            start_time = timeit.default_timer()
            matched_steps = match_ngrams_to_steps(texts[i], n_max, print_stats)
            end_time = timeit.default_timer()

            if print_stats:
                print(f"Time taken for {i}-th text: {end_time - start_time} seconds")
            full_matches.append(matched_steps)
            
        return full_matches
        
class SimpleNGramIndex(NGramIndex):
    def get_training_steps(self, input_ids):
        return  [info['idx'] for info in self.index[input_ids]]
    
    def train_index(self, texts, n_max, save_path=None):
        self.index = defaultdict(list)
        self.num_docs = len(texts)
        for n in range(1, n_max + 1):
            for idx, text in enumerate(texts):
                tokens = self.tokenizer.encode(text)
                for pos in range(len(tokens) - n):
                    kgram = tuple(tokens[pos:pos+n])
                    kgram_dict = {
                        "idx": idx,
                        "pos": pos,
                        "next_token": tokens[pos+n],
                    }
                    self.index[kgram].append(kgram_dict)


class InfiniGramIndex(NGramIndex):
    def load_index(self, index_path, **index_kwargs):
        self.index = InfiniGramEngine(index_path, **index_kwargs)
        self.num_docs = self.index.get_total_doc_cnt() # TODO: make sure this is correct

    @timeout(0.01)
    def get_training_steps(self,input_ids):
        results = self.index.find(input_ids=input_ids)
        segments = results['segment_by_shard']
        all_steps = []
        # https://infini-gram.readthedocs.io/en/latest/pkg.html#search-with-simple-queries
        for shard, rank_range in enumerate(segments):
            for rank in range(*rank_range):
                docs = self.index.get_doc_by_rank(s=shard, rank=rank, max_disp_len=10)
                metadata = json.loads(docs['metadata'])
                all_steps.append(metadata['step'])
        return all_steps
