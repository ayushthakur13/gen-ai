from datasets import DatasetDict

from src.prompting import build_qa_prompt


def format_qa_text(example):
    prompt = build_qa_prompt(example["question"])
    return {
        "prompt": prompt,
        "answer": example["answer"],
        "text": f"{prompt} {example['answer']}",
    }


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = 256) -> DatasetDict:
    formatted = dataset_dict.map(format_qa_text)

    def _tokenize(example):
        prompt = example["prompt"]
        answer = example["answer"]

        full_text = f"{prompt} {answer}{tokenizer.eos_token}"

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        labels = full_tokens["input_ids"].copy()
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))

        # Supervise answer generation only; ignore prompt and pad tokens in loss.
        for i in range(prompt_len):
            labels[i] = -100
        for i, token_id in enumerate(full_tokens["input_ids"]):
            if token_id == tokenizer.pad_token_id:
                labels[i] = -100

        full_tokens["labels"] = labels
        return full_tokens

    tokenized = formatted.map(_tokenize, remove_columns=formatted["train"].column_names)
    return tokenized
