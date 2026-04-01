
import requests
import time

def generate_answer(query, context, model):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    start = time.time()

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )

    latency = time.time() - start
    answer = res.json()["response"]

    tokens = len(prompt.split()) + len(answer.split())
    cost = tokens * 0.000001

    return answer, latency, cost
