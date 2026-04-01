from data_loader import load_msmarco, load_hotpotqa
from retrieval import HybridRetriever, ChromaDenseRetriever
from llm import generate_answer
from evaluation import exact_match, f1_score, recall_at_k
from faithfulness import judge_faithfulness
from stats import compute_stats
from tracker import init, log
from config import *

from tqdm import tqdm


def run():
    init()

    data1, corpus1 = load_msmarco()
    data2, corpus2 = load_hotpotqa()

    data = data1 + data2
    corpus = corpus1 + corpus2

    retrievers = {
        "dense": ChromaDenseRetriever(corpus, EMBEDDING_MODEL, CHROMA_PATH),
        "hybrid": HybridRetriever(corpus, EMBEDDING_MODEL, CHROMA_PATH, HYBRID_ALPHA)
    }

    for r_name, retriever in retrievers.items():
        for llm in LLM_CONFIGS:

            metrics = {
                "em": [],
                "f1": [],
                "recall": [],
                "faithfulness": [],
                "latency": [],
                "cost": []
            }

            for query, truth in tqdm(data):

                docs = retriever.search(query, TOP_K)
                context = "\n".join(docs)

                pred, latency, cost = generate_answer(query, context, llm["name"])

                metrics["em"].append(exact_match(pred, truth))
                metrics["f1"].append(f1_score(pred, truth))
                metrics["recall"].append(recall_at_k(docs, truth))
                metrics["faithfulness"].append(
                    judge_faithfulness(query, context, pred)
                )
                metrics["latency"].append(latency)
                metrics["cost"].append(cost)

            summary = {k: compute_stats(v) for k, v in metrics.items()}

            log({
                "retriever": r_name,
                "llm": llm["name"],
                **{f"{k}_mean": v["mean"] for k, v in summary.items()}
            })

            print(r_name, llm["name"], summary)


if __name__ == "__main__":
    run()