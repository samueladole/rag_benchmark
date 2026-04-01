
from datasets import load_dataset
import random

def load_msmarco():
    dataset = load_dataset("ms_marco", "v1.1", split="train[:500]")
    data, corpus = [], []

    for item in dataset:
        if item["passages"]["passage_text"]:
            query = item["query"]
            passages = item["passages"]["passage_text"]
            answer = passages[0]

            data.append((query, answer))
            corpus.extend(passages)

    return random.sample(data, 200), list(set(corpus))
