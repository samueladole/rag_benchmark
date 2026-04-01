import numpy as np

def compute_stats(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std