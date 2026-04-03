import os

from src.data_loader import create_dataset_splits
from src.evaluate import evaluate_models
from src.inference import load_base_model, load_fine_tuned_model
from src.preprocess import tokenize_dataset
from src.train import TrainConfig, train_lora_model


def print_comparison_table(metrics_rows):
    print("\n| Metric        | Base Model | Fine-tuned Model |")
    print("| ------------- | ---------- | ---------------- |")
    for row in metrics_rows:
        print(
            f"| {row['Metric']:<13} | {row['Base Model']:<10} | {row['Fine-tuned Model']:<16} |"
        )


def print_sample_comparisons(samples):
    print("\n=== SAMPLE COMPARISONS ===")
    for idx, row in enumerate(samples[:5], start=1):
        print(f"\nSample {idx}")
        print(f"Question      : {row['question']}")
        print(f"Base Output   : {row['base_prediction']}")
        print(f"Fine-tuned Out: {row['finetuned_prediction']}")
        print(f"Ground Truth  : {row['ground_truth']}")


def print_limitations():
    print("\n=== LIMITATIONS ===")
    print("1. Small curated + synthetic hybrid dataset creates overfitting risk.")
    print("2. Generative metrics (BLEU/ROUGE) are useful but not fully reliable for factual correctness.")


def main():
    assignment_root = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(assignment_root, "data")
    processed_dir = os.path.join(data_dir, "processed")
    models_dir = os.path.join(assignment_root, "models")
    outputs_dir = os.path.join(assignment_root, "outputs")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "dataset.json")

    print("\n[1/6] Creating curated + synthetic hybrid dataset and 80/10/10 splits...")
    dataset_dict = create_dataset_splits(
        dataset_path=dataset_path,
        processed_dir=processed_dir,
        total_samples=1100,
        force_regenerate=False,
    )

    print(
        f"Train: {len(dataset_dict['train'])} | Validation: {len(dataset_dict['validation'])} | "
        f"Test: {len(dataset_dict['test'])}"
    )

    print("\n[2/6] Tokenizing dataset for causal language modeling...")
    train_config = TrainConfig(
        model_name="distilgpt2",
        output_dir=models_dir,
        logs_dir=os.path.join(outputs_dir, "logs"),
        learning_rate=2e-5,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=256,
    )

    base_model, tokenizer = load_base_model(train_config.model_name)
    tokenized = tokenize_dataset(dataset_dict, tokenizer, max_length=train_config.max_length)

    print("\n[3/6] Fine-tuning with LoRA adapters (PEFT)...")
    train_output = train_lora_model(tokenized, train_config)

    print("\n[4/6] Loading base and fine-tuned models for evaluation...")
    base_model, base_tokenizer = load_base_model(train_output["base_model_name"])
    finetuned_model, finetuned_tokenizer = load_fine_tuned_model(
        adapter_dir=train_output["adapter_dir"],
        base_model_name=train_output["base_model_name"],
    )

    generation_config = {
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }

    print("\n[5/6] Evaluating base vs fine-tuned model on held-out test set...")
    evaluation = evaluate_models(
        test_samples=dataset_dict["test"],
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        finetuned_model=finetuned_model,
        finetuned_tokenizer=finetuned_tokenizer,
        output_dir=outputs_dir,
        generation_config=generation_config,
    )

    print("\n[6/6] Printing final report...")
    print_comparison_table(evaluation["metrics_rows"])
    print_sample_comparisons(evaluation["samples"])
    print_limitations()

    print("\nOutputs saved in outputs/:")
    print("- predictions.csv")
    print("- metrics_comparison.csv")
    print("- human_eval_base.csv")
    print("- human_eval_finetuned.csv")
    print("- sample_comparisons.csv")


if __name__ == "__main__":
    main()
