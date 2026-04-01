import requests

def judge_faithfulness(question, context, answer):
    prompt = f"""
Is the answer supported by the context?

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond with only YES or NO.
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )

    output = res.json()["response"].strip().lower()

    return 1 if "yes" in output else 0