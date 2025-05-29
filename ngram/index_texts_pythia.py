"""
What I did: 
- run the current main function which uses the function correlate_text_collect
- this saves a np array of texts |-> indices of all the n-grams to args.save
- give the texts path in args.texts_path
- then you can load this array to run different methods on the n-gram indicies 
"""

import argparse
import json
import numpy as np
import copy 

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

import random 
import scipy
import pickle
import timeit

import signal
from functools import wraps

TOTAL_STEPS = 100000

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def timeout(seconds):
    """
    Decorator: allow fn to run up to `seconds` (float), else return None.
    Unix-only, main thread.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            # 1) install our handler
            def _handler(signum, frame):
                raise TimeoutError

            old_handler = signal.signal(signal.SIGALRM, _handler)
            # 2) arm fractional‐second timer
            old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                return fn(*args, **kwargs)
            except TimeoutError:
                return []
            finally:
                # 3) disarm and restore
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                    signal.setitimer(signal.ITIMER_REAL, *old_timer)
                except Exception as e:
                    print(f"Error: {e}")
                    pass
        return wrapped
    return decorator


@timeout(0.01)
def get_training_steps(input_ids, batch_size, engine):
  results = engine.find(input_ids=input_ids)
  segments = results['segment_by_shard']
  all_steps = []
  # https://infini-gram.readthedocs.io/en/latest/pkg.html#search-with-simple-queries
  for shard, rank_range in enumerate(segments):
    for rank in range(*rank_range):
      docs = engine.get_doc_by_rank(s=shard, rank=rank, max_disp_len=10)
      metadata = json.loads(docs['metadata'])
      all_steps.append(metadata['step'] // batch_size)
  return all_steps

# from rohith
def correlate(tokens,k_max,batch_size,engine,tokenizer,print_stats=False):
    results = []
    for pos in range(len(tokens)-k_max):
        k = k_max
        while k > 0:
            kgram = tokens[pos:pos+k+1]
            start_time = timeit.default_timer()
            steps = get_training_steps(kgram,batch_size,engine)
            end_time = timeit.default_timer()
            if print_stats:
                print("ENGINE STATS:")
                print(f"Looked for the following {k}-gram: {repr(tokenizer.decode(kgram))}")
                print(f"Found {len(steps)} matches in {end_time - start_time} seconds")

            if len(steps) > 0:
                # random.shuffle(steps)
                # results.append(steps[0])
                results.append(steps)
                # results.append(int(np.mean(steps)))
                # results.append(np.mean(steps))
                break
            k -= 1
                    
    return results

# saves all the indices in a big list 
def correlate_unravel(tokens,k_max,batch_size,engine,tokenizer,print_stats=False):
    results = []
    for pos in range(len(tokens)-k_max):
        k = k_max
        temp = []
        while k > 0:
            kgram = tokens[pos:pos+k+1]
            start_time = timeit.default_timer()
            steps = get_training_steps(kgram,batch_size,engine)
            end_time = timeit.default_timer()
            if print_stats:
                print("ENGINE STATS:")
                print(f"Looked for the following {k}-gram: {repr(tokenizer.decode(kgram))}")
                print(f"Found {len(steps)} matches in {end_time - start_time} seconds")

            if len(steps) > 0:
                # random.shuffle(steps)
                # results.append(steps[0])
                temp.append(steps)
                # results.append(int(np.mean(steps)))
                # results.append(np.mean(steps))
                break
            k -= 1
        results.append(temp)
                    
    return results

def correlate_text_unravel(texts,tokenizer,k,batch_size,engine):
    full_results = []
    for i in range(len(texts)):
        start_time = timeit.default_timer()
        tokens = tokenizer.encode(texts[i])
        results = correlate_unravel(tokens=tokens,k_max=k,batch_size=batch_size,engine=engine,tokenizer=tokenizer)
        end_time = timeit.default_timer()
        print(f"Time taken for {i}-th text: {end_time - start_time} seconds")
        
        full_results.append(results)
        
    return full_results

def correlate_text(texts,tokenizer,k,batch_size,engine):
    full_results = []
    for i in range(len(texts)):
        start_time = timeit.default_timer()
        tokens = tokenizer.encode(texts[i])
        results = correlate(tokens=tokens,k_max=k,batch_size=batch_size,engine=engine,tokenizer=tokenizer)
        end_time = timeit.default_timer()
        print(f"Time taken for {i}-th text: {end_time - start_time} seconds")
        
        full_results.append(results)
        
    return full_results

# computes a z-score: shuffles the training order 'reps' times and takes a z-score over the means 
def correlate_text_zscore(texts,tokenizer,k,batch_size,engine,reps): 
    original_results = correlate_text(texts,tokenizer,k,batch_size,engine)
    original_results = flatten_list(original_results)

    mean_unshuffled = np.mean(original_results)
    # print(original_results)
    # print(len(original_results))

    means_shuffled = []
    for rep in range(reps):
        shuffle_order = np.arange(TOTAL_STEPS)
        random.shuffle(shuffle_order)

        shuffled_results = [shuffle_order[step] for step in original_results]
        means_shuffled.append(np.mean(shuffled_results))
    
    z_score = (mean_unshuffled - np.mean(means_shuffled)) / np.std(means_shuffled)
    p_value = scipy.stats.norm.cdf(-abs(z_score))

    print(f"all reps p-value: {p_value}")

    for n_reps in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        z_score = (mean_unshuffled - np.mean(means_shuffled[:n_reps])) / np.std(means_shuffled[:n_reps])
        p_value = scipy.stats.norm.cdf(-abs(z_score))

        print(f"{n_reps} p-value: {p_value}")

# permutation test: shuffles the training order 'reps' times
def correlate_text_permtest(texts,tokenizer,k,batch_size,engine,reps,save): 
    original_results = correlate_text(texts,tokenizer,k,batch_size,engine)
    original_results_unflat = copy.deepcopy(original_results)
    original_results = flatten_list(original_results)
    print(original_results)
    print(len(original_results))

    mean_unshuffled = np.mean(original_results)

    means_shuffled = []
    for rep in range(reps):
        shuffle_order = np.arange(TOTAL_STEPS)
        random.shuffle(shuffle_order)

        shuffled_results = [shuffle_order[step] for step in original_results]
        means_shuffled.append(np.mean(shuffled_results))

    for n_reps in [10, 20, 50, 100, 200, 500, 1000]:
        frac = sum(1 for x in means_shuffled[:n_reps] if x > mean_unshuffled)
        print(f"{n_reps}reps permtest p-value: {frac / n_reps}")
        z_score = (mean_unshuffled - np.mean(means_shuffled[:n_reps])) / np.std(means_shuffled[:n_reps])
        p_value = scipy.stats.norm.cdf(-abs(z_score))
        print(f"{n_reps}reps z-score p-value: {p_value}")

    with open(f'/nlp/scr/salzhu/{save}.pkl', "wb") as file:
        pickle.dump(original_results_unflat, file)
    print(original_results_unflat)

# saves all the indices in a big list 
def correlate_text_collect(texts,tokenizer,k,batch_size,engine,reps,save): 
    original_results = correlate_text_unravel(texts,tokenizer,k,batch_size,engine)
    original_results_unflat = copy.deepcopy(original_results)
    original_results = flatten_list(original_results)
    # print(original_results_unflat)
    # print(len(original_results_unflat))

    with open(f'/nlp/scr/salzhu/{save}.pkl', "wb") as file:
        pickle.dump(original_results_unflat, file)

# permutation test: shuffles the training order 'reps' times
# tests different (fixed) parameters like number of texts and number of repetitions 
def correlate_text_permtest_params(texts,tokenizer,k,batch_size,engine,reps): 
    original_results = correlate_text(texts,tokenizer,k,batch_size,engine)
    original_results_unflat = copy.deepcopy(original_results)
    original_results = flatten_list(original_results)

    mean_unshuffled = np.mean(original_results)

    print('orig')
    means_shuffled = []
    for rep in range(reps):
        shuffle_order = np.arange(TOTAL_STEPS)
        random.shuffle(shuffle_order)

        shuffled_results = [shuffle_order[step] for step in original_results]
        means_shuffled.append(np.mean(shuffled_results))

    for n_reps in [10, 20, 50, 100, 200, 500, 1000]:
        z_score = (mean_unshuffled - np.mean(means_shuffled[:n_reps])) / np.std(means_shuffled[:n_reps])
        p_value = scipy.stats.norm.cdf(-abs(z_score))

        print(f"orig zscore {n_reps} p-value: {p_value}")

    print('-----------------------------------')

    # print(original_results_unflat)
    # print(len(original_results_unflat))

    for n_texts in [50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]:

        means_shuffled = []
        for rep in range(reps):
            shuffle_order = np.arange(TOTAL_STEPS)
            random.shuffle(shuffle_order)

            original_results_temp = flatten_list(original_results_unflat[:n_texts])

            shuffled_results = [shuffle_order[step] for step in original_results_temp]
            means_shuffled.append(np.mean(shuffled_results))

        for n_reps in [10, 20, 50, 100, 200, 500, 1000]:
            frac = sum(1 for x in means_shuffled[:n_reps] if x > mean_unshuffled)
            print(f"{n_reps}reps {n_texts}texts permtest p-value: {frac / n_reps}")

            z_score = (mean_unshuffled - np.mean(means_shuffled[:n_reps])) / np.std(means_shuffled[:n_reps])
            p_value = scipy.stats.norm.cdf(-abs(z_score))
            print(f"{n_reps}reps {n_texts}texts zscore p-value: {p_value}")

        print()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--infinigram_index_dir', type=str, default='/scr-ssd/rohithk/data/index')
  parser.add_argument('--tokenizer_name',
                      type=str,
                      default='EleutherAI/pythia-6.9b-deduped',
                      help='The name of the tokenizer used to build the index.')
  parser.add_argument('--pretrain_batch_size', type=int, default=1024)
  parser.add_argument('--k', type=int, default=8)
  parser.add_argument('--save', type=str, default='test')
  parser.add_argument('--reps', type=int, default=100)
  parser.add_argument('--texts_path', type=str, default='/nlp/scr/rohithk/data/pythia_6.9b_step100000_pile_0.pkl')
  parser.add_argument('--n_texts', type=int, default=5000)
  parser.add_argument('--start_text', type=int, default=0)

  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
  engine = InfiniGramEngine(index_dir=args.infinigram_index_dir,
                            eos_token_id=tokenizer.eos_token_id)
  
  texts = pickle.load(open(args.texts_path, 'rb'))
  texts = texts[args.start_text:args.start_text+args.n_texts]
  
  # correlate_text_zscore(texts,tokenizer,args.k,args.pretrain_batch_size,engine,args.reps)
  # correlate_text_permtest(texts,tokenizer,args.k,args.pretrain_batch_size,engine,args.reps,args.save)
  correlate_text_collect(texts,tokenizer,args.k,args.pretrain_batch_size,engine,args.reps,args.save)
  # correlate_text_permtest_params(texts,tokenizer,args.k,args.pretrain_batch_size,engine,args.reps)
  
