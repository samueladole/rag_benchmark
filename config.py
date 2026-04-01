NUM_SAMPLES = 300
TOP_K = 5
CHROMA_PATH = "./chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HYBRID_ALPHA = 0.6  # weight for dense vs BM25

LLM_CONFIGS = [
    {"name": "llama3"},
    {"name": "mistral"},
]