import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(query, context, model):
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    latency = time.time() - start

    result = response.json()
    answer = result["response"]

    # Rough cost proxy (since local models don't charge)
    tokens_estimate = len(prompt.split()) + len(answer.split())
    cost = tokens_estimate * 0.000001

    return answer, latency, cost