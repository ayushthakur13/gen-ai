from datasets import DatasetDict

from src.prompting import build_qa_prompt


def format_qa_text(example):
    prompt = build_qa_prompt(example["question"])
    return {
        "input_text": prompt,
        "target_text": example["answer"],
    }


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = 256) -> DatasetDict:
    formatted = dataset_dict.map(format_qa_text)

    def _tokenize(example):
        model_inputs = tokenizer(
            example["input_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        label_tokens = tokenizer(
            text_target=example["target_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = label_tokens["input_ids"]
        model_inputs["labels"] = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in labels
        ]
        return model_inputs

    tokenized = formatted.map(_tokenize, remove_columns=formatted["train"].column_names)
    return tokenized
