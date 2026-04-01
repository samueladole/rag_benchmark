
import re

def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()

def exact_match(pred, truth):
    return normalize(pred) == normalize(truth)

def f1_score(pred, truth):
    p, t = normalize(pred).split(), normalize(truth).split()
    common = set(p) & set(t)
    if not common: return 0
    precision = len(common)/len(p)
    recall = len(common)/len(t)
    return 2 * precision * recall / (precision + recall)

def recall_at_k(retrieved_docs, ground_truth):
    for doc in retrieved_docs:
        if ground_truth.lower() in doc.lower():
            return 1
    return 0
