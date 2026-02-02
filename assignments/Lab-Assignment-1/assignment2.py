from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq()

model = "openai/gpt-oss-120b"

prompts = [
    "What is Machine Learning? Answer in 2 lines.",
    "What is Machine Learning? Explain using a real-life example.",
    "What is Machine Learning? Explain for a 10-year-old child."
]

for p in prompts:
    print(f"\nPROMPT: {p}\n" + "-" * 50)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": p}
        ],
        temperature=1,
        max_completion_tokens=256
    )

    print(completion.choices[0].message.content)
