
import numpy as np

def compute_stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}
