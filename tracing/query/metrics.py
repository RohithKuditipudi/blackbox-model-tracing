import numpy as np
import evaluate

def eval_model(model_path, texts, metric):
    stats = metric(model_path,texts)

    return stats

def pplx(model_path, texts):
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=texts)
    pplx = np.log(result['perplexities'])

    return pplx