import csv
import os
import random
from typing import Dict, List

from src.inference import generate_answer
from src.metrics import compute_metrics, manual_human_evaluation


def _write_predictions_csv(rows: List[Dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "ground_truth", "base_prediction", "finetuned_prediction"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_csv(metrics_rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Base Model", "Fine-tuned Model"])
        writer.writeheader()
        writer.writerows(metrics_rows)


def _write_human_eval_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "prediction", "ground_truth", "score", "rationale"],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_models(
    test_samples,
    base_model,
    base_tokenizer,
    finetuned_model,
    finetuned_tokenizer,
    output_dir: str,
    generation_config: Dict[str, float],
) -> Dict[str, object]:
    questions = test_samples["question"]
    references = test_samples["answer"]

    base_predictions = []
    finetuned_predictions = []

    for question in questions:
        base_pred = generate_answer(base_model, base_tokenizer, question, generation_config)
        ft_pred = generate_answer(finetuned_model, finetuned_tokenizer, question, generation_config)

        base_predictions.append(base_pred)
        finetuned_predictions.append(ft_pred)

    prediction_rows = []
    for q, gt, bp, fp in zip(questions, references, base_predictions, finetuned_predictions):
        prediction_rows.append(
            {
                "question": q,
                "ground_truth": gt,
                "base_prediction": bp,
                "finetuned_prediction": fp,
            }
        )

    _write_predictions_csv(prediction_rows, os.path.join(output_dir, "predictions.csv"))

    base_metrics = compute_metrics(base_predictions, references)
    finetuned_metrics = compute_metrics(finetuned_predictions, references)

    base_human = manual_human_evaluation(questions, base_predictions, references, sample_count=8)
    finetuned_human = manual_human_evaluation(questions, finetuned_predictions, references, sample_count=8)

    base_metrics["Proxy Human Score"] = base_human["average_score"]
    finetuned_metrics["Proxy Human Score"] = finetuned_human["average_score"]

    metrics_rows = []
    ordered_metrics = ["Exact Match", "BLEU", "ROUGE-L", "Keyword Score", "Proxy Human Score"]

    for metric_name in ordered_metrics:
        metrics_rows.append(
            {
                "Metric": metric_name,
                "Base Model": round(base_metrics[metric_name], 4),
                "Fine-tuned Model": round(finetuned_metrics[metric_name], 4),
            }
        )

    _write_metrics_csv(metrics_rows, os.path.join(output_dir, "metrics_comparison.csv"))

    _write_human_eval_csv(
        base_human["details"],
        os.path.join(output_dir, "human_eval_base.csv"),
    )
    _write_human_eval_csv(
        finetuned_human["details"],
        os.path.join(output_dir, "human_eval_finetuned.csv"),
    )

    sample_rng = random.Random(42)
    sample_count = min(8, len(prediction_rows))
    sample_indices = sorted(sample_rng.sample(range(len(prediction_rows)), sample_count))
    sample_rows = [prediction_rows[i] for i in sample_indices]
    _write_predictions_csv(sample_rows, os.path.join(output_dir, "sample_comparisons.csv"))

    return {
        "metrics_rows": metrics_rows,
        "samples": sample_rows,
    }
