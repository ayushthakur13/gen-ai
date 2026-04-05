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
Structured template-based QA dataset with a small curated core.

### Construction Strategy
1. Curated base from a domain concept knowledge bank (`concept`, `definition`, `example`, `contrast`).
2. An upgraded high-quality slice (~80 samples) with stronger technical phrasing and example-grounded answers.
3. Deterministic template expansion for coverage across the concept set.
4. Short factual answers built from the knowledge bank to keep supervision clean.

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
2. Selected model: `google/flan-t5-small`

Rationale:
1. Instruction-tuned base model that is better aligned with QA.
2. Lightweight enough for local LoRA fine-tuning.

### Fine-Tuning Method
LoRA via PEFT on a seq2seq model.

LoRA configuration:
1. Task type: `SEQ_2_SEQ_LM`
2. Rank (`r`): 16
3. `lora_alpha`: 32
4. `lora_dropout`: 0.05
5. Target modules: `q`, `v`

Trainable parameter ratio in runs is approximately 0.89%, confirming parameter-efficient adaptation.

## Preprocessing and Training Pipeline

### Prompt Format
Training text uses:
1. `Answer the question concisely using correct technical terminology and include a specific example if relevant.`
2. `Question: {question}`
3. `Answer: {answer}`

### Tokenization
1. Max sequence length: 256
2. Truncation: enabled
3. Padding: `max_length`
4. Label masking: prompt and padding tokens masked to focus learning on answer tokens

### Training Configuration
1. Epochs: 5
2. Learning rate: `1e-4`
3. Per-device train batch size: 2
4. Per-device eval batch size: 2
5. Gradient accumulation: 4
6. Eval strategy: epoch
7. Save strategy: epoch
8. Best model metric: `eval_loss`

Latest validation-loss trend (epoch-wise): `3.444 -> 3.107 -> 2.929 -> 2.836 -> 2.807`.

## Inference and Evaluation Design

### Inference Setup
Both base and fine-tuned models are evaluated on the same held-out test set.

Controlled generation settings use beam search, a max token cap, and anti-repetition constraints to keep outputs stable for comparison.

### Metrics Implemented
1. BLEU
2. ROUGE-L
3. Keyword Overlap (task-specific metric)

## Latest Full-Run Results
From `outputs/metrics_comparison.csv`:

| Metric | Base Model | Fine-tuned Model |
| --- | ---: | ---: |
| BLEU | 0.0239 | 0.1369 |
| ROUGE-L | 0.2306 | 0.3367 |
| Keyword Score | 0.1610 | 0.2701 |

### Result Interpretation
1. Fine-tuned model improved on BLEU, ROUGE-L, and Keyword Score.
2. Keyword Score provides a task-specific signal for domain-term alignment.
3. Overall improvement is modest but measurable and technically defensible for assignment scope.

## Qualitative Findings
1. Fine-tuned outputs show better overlap with domain terms in many samples.
2. Some responses still remain generic or partially paraphrased.
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
3. `sample_comparisons.csv`

## Limitations
1. `google/flan-t5-small` is still compact compared with larger instruction-tuned models.
2. Overlap metrics do not fully capture factual correctness.
3. The dataset is intentionally template-driven, so answers can still sound formulaic.

## Conclusion
This assignment successfully delivers a full custom-data fine-tuning pipeline with LoRA-based PEFT and reproducible evaluation. The final run shows measurable gains over the base model on multiple metrics, validating partial domain adaptation under local compute constraints.