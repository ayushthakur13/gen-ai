import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    model_name: str = "distilgpt2"
    output_dir: str = "models"
    logs_dir: str = "outputs/logs"
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    seed: int = 42


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def train_lora_model(tokenized_datasets, config: TrainConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    base_model, tokenizer = load_base_model_and_tokenizer(config.model_name)
    peft_model = apply_lora(base_model)
    use_cpu = get_device() == "cpu"

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        logging_dir=config.logs_dir,
        report_to="none",
        bf16=False,
        fp16=False,
        seed=config.seed,
        dataloader_pin_memory=False,
        do_train=True,
        do_eval=True,
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    adapter_dir = os.path.join(config.output_dir, "lora_adapter")
    peft_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return {
        "adapter_dir": adapter_dir,
        "tokenizer": tokenizer,
        "base_model_name": config.model_name,
    }
