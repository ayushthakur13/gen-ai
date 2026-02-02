from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq()

model = "openai/gpt-oss-120b"
prompt = "List three applications of Artificial Intelligence in daily life."

for i in range(1, 4):
    print(f"\nRUN {i}\n" + "-" * 50)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_completion_tokens=128
    )

    print(completion.choices[0].message.content)
