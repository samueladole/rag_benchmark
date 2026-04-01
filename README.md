# 🔍 RAG Benchmarking Framework

**Dense vs Hybrid Retrieval • LLaMA 3 vs Mistral • End-to-End Evaluation**

---

## 🚀 Overview

This project implements a **production-style benchmarking framework** for evaluating Retrieval-Augmented Generation (RAG) systems.

It compares:

* **Retrieval strategies** → Dense vs Hybrid (BM25 + embeddings)
* **LLM configurations** → LLaMA 3 vs Mistral
* **Performance dimensions** → Accuracy, Retrieval Quality, Faithfulness, Latency, and Cost

The system is evaluated using:

* MS MARCO (retrieval-focused QA)
* HotpotQA (multi-hop reasoning)

---

## 🎯 Key Objectives

* Build a **modular RAG pipeline**
* Compare **retrieval strategies under identical conditions**
* Measure how retrieval quality affects **LLM output accuracy**
* Introduce **faithfulness evaluation (LLM-as-judge)**
* Provide **statistically grounded insights**

---

## 🧠 Key Features

### 🔎 Retrieval Layer

* **Dense Retrieval**

  * ChromaDB vector store
  * Sentence-transformer embeddings
* **Hybrid Retrieval**

  * BM25 (lexical search)
  * Dense embeddings
  * Weighted score fusion

---

### 🤖 LLM Layer

* Local inference via Ollama:

  * LLaMA 3
  * Mistral

* Controlled generation setup:

  * Same prompts
  * Same retrieved context
  * Fair comparison across models

---

### 📊 Evaluation Metrics

| Category    | Metric           | Purpose                           |
| ----------- | ---------------- | --------------------------------- |
| Accuracy    | Exact Match (EM) | Strict correctness                |
| Accuracy    | F1 Score         | Partial correctness               |
| Retrieval   | Recall@K         | Did we retrieve relevant context? |
| Reliability | Faithfulness     | Is answer grounded in context?    |
| Performance | Latency          | Response time                     |
| Efficiency  | Cost (proxy)     | Token usage estimate              |

---

## 🏗️ System Architecture

```
                ┌────────────────────┐
                │    User Query      │
                └────────┬───────────┘
                         │
          ┌──────────────┴──────────────┐
          │        Retrieval Layer       │
          │                              │
   ┌──────────────┐          ┌──────────────┐
   │ Dense Search │          │ Hybrid Search│
   │ (ChromaDB)   │          │ (BM25 + Dense)│
   └──────┬───────┘          └──────┬───────┘
          │                         │
          └──────────┬──────────────┘
                     │
              Top-K Context
                     │
              ┌──────▼──────┐
              │    LLMs     │
              │ LLaMA 3     │
              │ Mistral     │
              └──────┬──────┘
                     │
              Generated Answer
                     │
        ┌────────────┴────────────┐
        │     Evaluation Layer     │
        │ EM | F1 | Recall@K       │
        │ Faithfulness | Latency   │
        └────────────┬────────────┘
                     │
              Experiment Tracking
```

---

## 📂 Project Structure

```
rag-benchmark/
├── chroma_db/            # Persistent vector database
├── data_loader.py        # Dataset ingestion (MS MARCO, HotpotQA)
├── retrieval.py          # Dense + Hybrid retrieval logic
├── llm.py                # LLM inference (Ollama)
├── evaluation.py         # Metrics (EM, F1, Recall@K)
├── faithfulness.py       # LLM-as-judge evaluation
├── stats.py              # Statistical analysis
├── tracker.py            # Experiment tracking (W&B)
├── config.py             # Configurations
├── main.py               # Pipeline entry point
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-benchmark.git
cd rag-benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama + Models

```bash
ollama pull llama3
ollama pull mistral
```

### 4. Run Benchmark

```bash
python main.py
```

---

## 📊 Example Results

| Retriever | Model   | F1   | Recall@K | Faithfulness | Latency |
| --------- | ------- | ---- | -------- | ------------ | ------- |
| Dense     | LLaMA3  | 0.42 | 0.65     | 0.78         | 1.2s    |
| Hybrid    | Mistral | 0.48 | 0.72     | 0.81         | 1.4s    |

---

## 🔬 Key Insights

* **Hybrid retrieval improves Recall@K**, leading to better answers
* Retrieval quality has **greater impact than LLM choice**
* Higher recall correlates with improved **faithfulness**
* Trade-offs exist between **latency and accuracy**

---

## ⚡ Design Decisions

* **ChromaDB** for lightweight, persistent vector storage
* **BM25 + Dense fusion** for robust retrieval
* **Local LLMs** for cost-free experimentation
* **LLM-as-judge** for scalable evaluation

---

## 🧪 Limitations

* Cost is estimated (not real API billing)
* Faithfulness depends on LLM judgment reliability
* No reranking (yet)

---

## 🚀 Future Improvements

* Cross-encoder reranking (e.g., MiniLM reranker)
* Query decomposition for multi-hop QA
* Experiment dashboard (Streamlit / Next.js)
* Distributed benchmarking

---

## 💼 Impact

This project demonstrates:

* Applied **RAG system design**
* **Retrieval + LLM integration**
* **Evaluation methodology (research-level)**
* Production-ready **ML system thinking**

---

## 👨‍💻 Author

**Samuel Adole**

---

## ⭐ If You Like This Project

Give it a star ⭐ and feel free to contribute!
