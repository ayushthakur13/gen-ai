import time
from groq import Groq
from src.config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

def call_llm(prompt):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    latency = time.time() - start

    return {
        "output": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency": latency
    }
