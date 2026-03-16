import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# 1. Load Dataset
dataset = load_dataset("Abirate/english_quotes")

print("\nDataset Example:")
print(dataset["train"][0])


# 2. Load Tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT models don't have pad token by default
tokenizer.pad_token = tokenizer.eos_token


# 3. Tokenize Dataset
def tokenize_function(example):
    return tokenizer(
        example["quote"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)


# 4. Load Pretrained Model
model = AutoModelForCausalLM.from_pretrained(model_name)


# 5. Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    report_to="none"
)


# 6. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)


# 8. Fine-tune Model
print("\nStarting training...\n")
trainer.train()


# 9. Text Generation (Inference)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

prompt = "Success is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=40
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== GENERATED TEXT ===\n")
print(generated_text)