# Assignment-2: Fine-Tuning a Large Language Model Using Custom Dataset

## Objective
The objective of this assignment is to understand how fine-tuning improves the performance of pretrained large language models for specialized tasks by adapting them to domain-specific datasets.

This project implements a complete end-to-end pipeline for domain-specific QA in AI/ML/LLM concepts using Hugging Face Transformers and PEFT LoRA.

## Problem Statement
Pretrained LLMs are trained on broad, general-purpose corpora, so their responses on technical QA tasks are often generic, noisy, or weakly grounded. Fine-tuning on curated task-specific data helps the model learn expected terminology, structure, and answer style.

This assignment investigates whether LoRA-based adaptation of a GPT-style open model improves performance over the base model.

## Task Selection
Chosen task: Domain-specific question answering.

Domain: Artificial Intelligence, Machine Learning, and LLM concepts.

Task I/O:
1. Input: technical question
2. Output: concise factual answer

## End-to-End Lifecycle Covered
1. Model selection
2. Dataset creation and validation
3. Preprocessing and tokenization
4. Parameter-efficient fine-tuning (LoRA)
5. Base and fine-tuned inference
6. Multi-metric evaluation and comparison

## Dataset Methodology

### Dataset Type
Curated + synthetic hybrid dataset.

### Construction Strategy
1. Curated base from a domain concept knowledge bank (`concept`, `definition`, `example`, `contrast`).
2. Synthetic expansion via template-based question generation with style/context variations.
3. Controlled answer variants to improve diversity:
	- 1 sentence
	- 2 sentences
	- 3 sentences

### Quality Controls
1. Duplicate question filtering with normalization.
2. Schema checks (`question`, `answer` non-empty strings).
3. Automatic regeneration if dataset file is missing, malformed, or wrong size.

### Final Dataset Size and Split
1. Total samples: 1100
2. Train: 880 (80%)
3. Validation: 110 (10%)
4. Test: 110 (10%)

This satisfies assignment requirements (500-2000 examples).

## Model and Fine-Tuning Approach

### Base Model
1. Model family: GPT-style open model
2. Selected model: `distilgpt2`

Rationale:
1. Lightweight and feasible for local training.
2. Compatible with causal LM objective and Hugging Face tooling.

### Fine-Tuning Method
LoRA via PEFT (no full-model fine-tuning).

LoRA configuration:
1. Task type: `CAUSAL_LM`
2. Rank (`r`): 8
3. `lora_alpha`: 16
4. `lora_dropout`: 0.1
5. Target modules: `c_attn`, `c_proj`

Trainable parameter ratio in runs is approximately 0.49%, confirming parameter-efficient adaptation.

## Preprocessing and Training Pipeline

### Prompt Format
Training text uses:
1. `You are an AI/ML assistant. Provide concise factual answers.`
2. `Question: {question}`
3. `Answer: {answer}`

### Tokenization
1. Max sequence length: 256
2. Truncation: enabled
3. Padding: `max_length`
4. Label masking: prompt and padding tokens masked to focus learning on answer tokens

### Training Configuration
1. Epochs: 2
2. Learning rate: `2e-5`
3. Per-device train batch size: 2
4. Per-device eval batch size: 2
5. Gradient accumulation: 4
6. Eval strategy: epoch
7. Save strategy: epoch
8. Best model metric: `eval_loss`

## Inference and Evaluation Design

### Inference Setup
Both base and fine-tuned models are evaluated on the same held-out test set.

Controlled generation settings are used with max token cap and anti-repetition constraints to keep outputs stable for comparison.

### Metrics Implemented
1. Exact Match
2. BLEU
3. ROUGE-L
4. Keyword Overlap
5. Proxy Human Score (assignment rubric-based proxy)

## Latest Full-Run Results
From `outputs/metrics_comparison.csv`:

| Metric | Base Model | Fine-tuned Model |
| --- | ---: | ---: |
| Exact Match | 0.0000 | 0.0000 |
| BLEU | 0.0094 | 0.0110 |
| ROUGE-L | 0.0897 | 0.1013 |
| Keyword Score | 0.0454 | 0.0650 |
| Proxy Human Score | 1.0000 | 1.0000 |

### Result Interpretation
1. Fine-tuned model improved on BLEU, ROUGE-L, and Keyword Score.
2. Exact Match remained unchanged (strict metric for generative outputs).
3. Proxy Human Score remained unchanged in this run.
4. Overall improvement is modest but measurable and technically defensible for assignment scope.

## Qualitative Findings
1. Fine-tuned outputs show better overlap with domain terms in many samples.
2. Some responses still remain generic or weakly grounded.
3. Performance indicates partial domain adaptation rather than robust expert-level QA behavior.

## Repository Structure
```text
assignment-2-fine-tune-llm/
├── data/
│   ├── dataset.json
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── metrics.py
├── models/
├── outputs/
└── main.py
```

## How to Run
From this directory:

```bash
python3 main.py
```

## Output Artifacts
Generated in `outputs/`:
1. `predictions.csv`
2. `metrics_comparison.csv`
3. `human_eval_base.csv`
4. `human_eval_finetuned.csv`
5. `sample_comparisons.csv`

## Limitations
1. `distilgpt2` capacity is limited for high-fidelity technical QA.
2. Overlap metrics do not fully capture factual correctness.
3. Proxy Human Score is rubric-based and not multi-rater manual evaluation.
4. Even with 1100 samples, synthetic-heavy data can bias style and phrasing.

## Conclusion
This assignment successfully delivers a full custom-data fine-tuning pipeline with LoRA-based PEFT and reproducible evaluation. The final run shows measurable gains over the base model on multiple metrics, validating partial domain adaptation under local compute constraints.