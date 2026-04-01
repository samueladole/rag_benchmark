
# 🔍 RAG Benchmarking Framework

Production-style evaluation framework for RAG systems comparing Dense vs Hybrid retrieval and LLMs (LLaMA3, Mistral).

## 🚀 Features
- ChromaDB vector store (persistent)
- Hybrid retrieval (BM25 + embeddings)
- Local LLMs via Ollama
- Metrics: EM, F1, Recall@K, Faithfulness, Latency, Cost
- Experiment tracking (Weights & Biases)

## ⚙️ Setup
pip install -r requirements.txt

Install Ollama:
ollama pull llama3
ollama pull mistral

Run:
python main.py
