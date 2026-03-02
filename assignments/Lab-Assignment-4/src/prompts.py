def zero_shot(text):
    return f"Classify sentiment as Positive or Negative:\nText: {text}"

def role_based(text):
    return f"""
You are a strict sentiment classifier.
Respond with only Positive or Negative.

Text: {text}
"""

def few_shot(text):
    return f"""
Text: I love this phone. → Positive
Text: This is terrible. → Negative
Text: {text} →
"""

def structured_output(text):
    return f"""
Classify sentiment and respond in JSON format:

{{
  "sentiment": "Positive or Negative"
}}

Text: {text}
"""
