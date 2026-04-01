
from data_loader import load_msmarco
from retrieval import ChromaDenseRetriever, HybridRetriever
from llm import generate_answer
from evaluation import exact_match, f1_score, recall_at_k
from stats import compute_stats
from config import *

def run():
    data, corpus = load_msmarco()

    retrievers = {
        "dense": ChromaDenseRetriever(corpus, EMBEDDING_MODEL, CHROMA_PATH),
        "hybrid": HybridRetriever(corpus, EMBEDDING_MODEL, CHROMA_PATH)
    }

    for name, retriever in retrievers.items():
        for llm in LLM_CONFIGS:

            metrics = {"em": [], "f1": [], "recall": [], "latency": [], "cost": []}

            for q, t in data:
                docs = retriever.search(q, TOP_K)
                context = "\n".join(docs)

                pred, lat, cost = generate_answer(q, context, llm["name"])

                metrics["em"].append(exact_match(pred, t))
                metrics["f1"].append(f1_score(pred, t))
                metrics["recall"].append(recall_at_k(docs, t))
                metrics["latency"].append(lat)
                metrics["cost"].append(cost)

            summary = {k: compute_stats(v) for k, v in metrics.items()}
            print(name, llm["name"], summary)

if __name__ == "__main__":
    run()
