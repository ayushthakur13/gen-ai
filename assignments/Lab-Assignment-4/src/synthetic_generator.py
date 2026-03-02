import csv
import json
from src.groq_client import call_llm

TARGET_PER_CLASS = 50
BATCH_SIZE = 10

def generate_batch(label, count):
    prompt = f"""
Generate {count} realistic product reviews.

All reviews must be labeled as {label}.

Respond in JSON list format:
[
  {{"text": "...", "label": "{label}"}}
]

Ensure diversity in tone and wording.
"""

    result = call_llm(prompt)

    try:
        data = json.loads(result["output"])
        return data
    except:
        print("JSON parsing failed. Retrying...")
        return []

def generate_dataset():
    positive_data = []
    negative_data = []

    while len(positive_data) < TARGET_PER_CLASS:
        batch = generate_batch("Positive", BATCH_SIZE)
        positive_data.extend(batch)

    while len(negative_data) < TARGET_PER_CLASS:
        batch = generate_batch("Negative", BATCH_SIZE)
        negative_data.extend(batch)

    # Trim to exact count
    positive_data = positive_data[:TARGET_PER_CLASS]
    negative_data = negative_data[:TARGET_PER_CLASS]

    final_data = positive_data + negative_data

    with open("data/synthetic_sentiment.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

        for item in final_data:
            writer.writerow([item["text"], item["label"]])

    print("Synthetic dataset generated successfully.")
