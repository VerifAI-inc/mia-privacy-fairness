import numpy as np

def uncertainty_select(probas_val, num_samples):
    rev = np.sort(probas_val, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    selection = np.argsort(values)[:num_samples]
    return selection

def entropy_select(probas_val, num_samples):
    e = (-probas_val * np.log2(probas_val)).sum(axis=1)
    selection = (np.argsort(e)[::-1])[:num_samples]
    return selection

def random_select(probas_val, num_samples):
    random_state = check_random_state(0)
    selection = np.random.choice(probas_val.shape[0], num_samples, replace=False)
    return selection

