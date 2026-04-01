
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class ChromaDenseRetriever:
    def __init__(self, texts, model_name, path):
        self.texts = texts
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("rag")

        if self.collection.count() == 0:
            embeddings = self.model.encode(texts).tolist()
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=[str(i) for i in range(len(texts))]
            )

    def search(self, query, k=5):
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        return results["documents"][0]


class HybridRetriever:
    def __init__(self, texts, model_name, path):
        self.texts = texts
        self.bm25 = BM25Okapi([t.split() for t in texts])
        self.dense = ChromaDenseRetriever(texts, model_name, path)

    def search(self, query, k=5):
        dense_docs = self.dense.search(query, k=20)
        scores = self.bm25.get_scores(query.split())

        ranked = sorted(
            dense_docs,
            key=lambda doc: scores[self.texts.index(doc)],
            reverse=True
        )

        return ranked[:k]
