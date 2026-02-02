from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq()

models = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile"
]

prompt = "Explain Artificial Intelligence to a first-year college student."

for model in models:
    print(f"\nMODEL: {model}\n" + "-" * 50)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_completion_tokens=512
    )

    print(completion.choices[0].message.content)
