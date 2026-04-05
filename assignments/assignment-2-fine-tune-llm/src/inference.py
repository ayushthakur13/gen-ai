import re
from typing import Dict

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.prompting import build_qa_prompt


def load_base_model(model_name: str = "google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_fine_tuned_model(adapter_dir: str, base_model_name: str = "google/flan-t5-small"):
    base_model, tokenizer = load_base_model(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.eval()
    return peft_model, tokenizer


def _extract_answer(decoded_output: str) -> str:
    text = decoded_output.strip()
    text = text.split("\nQuestion:", 1)[0].strip()
    text = text.split("\n", 1)[0].strip()

    sentence_split = re.split(r"(?<=[.!?])\s+", text)
    clipped = [s.strip() for s in sentence_split if s.strip()][:3]
    if clipped:
        return " ".join(clipped)

    return text


def generate_answer(
    model,
    tokenizer,
    question: str,
    generation_config: Dict[str, float],
) -> str:
    prompt = build_qa_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        do_sample = generation_config.get("do_sample", False)
        gen_kwargs = {
            "max_new_tokens": generation_config["max_new_tokens"],
            "do_sample": do_sample,
            "repetition_penalty": generation_config.get("repetition_penalty", 1.2),
            "no_repeat_ngram_size": generation_config.get("no_repeat_ngram_size", 3),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = generation_config.get("temperature", 0.7)
            gen_kwargs["top_k"] = generation_config.get("top_k", 50)
            gen_kwargs["top_p"] = generation_config.get("top_p", 0.9)
        else:
            gen_kwargs["num_beams"] = generation_config.get("num_beams", 4)

        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _extract_answer(decoded)
