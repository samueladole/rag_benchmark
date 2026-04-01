import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import os


class ChromaDenseRetriever:
    def __init__(self, texts, model_name, path):
        self.texts = texts
        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("rag")

        if self.collection.count() == 0:
            BATCH_SIZE = 1000 # Safe batch size for encoding

            embeddings = self.model.encode(texts).tolist()

            for i in range(0, len(texts), BATCH_SIZE):
                batch_docs = texts[i:i + BATCH_SIZE]
                batch_embs = self.model.encode(batch_docs).tolist()

                self.collection.add(
                    documents=batch_docs,
                    embeddings=batch_embs,
                    ids=[str(j) for j in range(i, i + len(batch_docs))]
                )

    def search(self, query, k=5):
        q_emb = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=k
        )

        docs = results["documents"][0]

        # Flatten if nested
        flattened = []
        for d in docs:
            if isinstance(d, list):
                flattened.extend(d)
            else:
                flattened.append(d)

        return flattened, results["distances"][0]


class HybridRetriever:
    def __init__(self, texts, model_name, path, alpha=0.5):
        self.texts = texts
        self.alpha = alpha

        self.tokenized = [t.split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

        self.dense = ChromaDenseRetriever(texts, model_name, path)

    def search(self, query, k=5):
        dense_docs, dense_scores = self.dense.search(query, k=20)

        dense_docs = [d for d in dense_docs if isinstance(d, str)]

        bm25_scores = self.bm25.get_scores(query.split())

        combined = []
        for doc, d_score in zip(dense_docs, dense_scores):
            bm25_score = bm25_scores[self.texts.index(doc)]

            score = self.alpha * (1 / (1 + d_score)) + (1 - self.alpha) * bm25_score
            combined.append((doc, score))

        ranked = sorted(combined, key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:k]]