import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)

# 1. TEXT GENERATION (GPT-2)
def text_generation():

    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Artificial Intelligence will transform the future by"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== TEXT GENERATION ===")
    print(generated_text)


# 2. SUMMARIZATION (T5)
def summarization():

    model_name = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = """
    Artificial Intelligence (AI) is a branch of computer science that focuses on
    creating intelligent machines capable of performing tasks that typically
    require human intelligence such as learning, reasoning, and problem solving.
    AI is widely used in recommendation systems, virtual assistants, healthcare,
    finance, and autonomous vehicles.
    """

    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=40,
        min_length=10
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== SUMMARIZATION ===")
    print(summary)


# 3. QUESTION ANSWERING (DistilBERT)
def question_answering():

    model_name = "distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    context = """
    Hugging Face is a company that develops tools for natural language processing.
    The Transformers library provides thousands of pretrained models for tasks
    such as text classification, summarization, translation, and question answering.
    """

    question = "What does the Transformers library provide?"

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt"
    )

    outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    answer_tokens = inputs["input_ids"][0][start_index:end_index]

    answer = tokenizer.decode(answer_tokens)

    print("\n=== QUESTION ANSWERING ===")
    print("Question:", question)
    print("Answer:", answer)


# RUN ALL TASKS
if __name__ == "__main__":

    text_generation()
    summarization()
    question_answering()
