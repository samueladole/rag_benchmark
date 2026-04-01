from datasets import load_dataset
import random

def load_msmarco():
    dataset = load_dataset("ms_marco", "v1.1", split="train[:1000]")

    data = []
    corpus = []

    for item in dataset:
        query = item["query"]

        if item["passages"]["passage_text"]:
            passages = item["passages"]["passage_text"]

            # take top passage as ground truth
            answer = passages[0]

            data.append((query, answer))

            for p in passages:
                corpus.append(p)

    return random.sample(data, 300), list(set(corpus))


def load_hotpotqa():
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train[:300]")

    data = []
    corpus = []

    for item in dataset:
        query = item["question"]
        answer = item["answer"]

        context = [" ".join(p[1]) for p in item["context"]]

        data.append((query, answer))
        corpus.extend(context)

    return data, list(set(corpus))