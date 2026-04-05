import json
import os
import random
from typing import Dict, List

from datasets import Dataset, DatasetDict


SEED = 42
RNG = random.Random(SEED)
UPGRADED_SAMPLE_COUNT = 80


CONCEPT_KB: List[Dict[str, str]] = [
    {
        "concept": "Artificial Intelligence",
        "definition": "the field of building systems that perform tasks requiring human-like intelligence",
        "example": "rule-based expert systems and modern learning-based agents",
        "contrast": "narrow AI focuses on one task, while AGI would generalize across tasks",
    },
    {
        "concept": "Machine Learning",
        "definition": "a subset of AI where models learn patterns from data instead of explicit rules",
        "example": "spam filtering and demand forecasting",
        "contrast": "traditional programming encodes logic manually",
    },
    {
        "concept": "Deep Learning",
        "definition": "a class of machine learning that uses multi-layer neural networks",
        "example": "image classification with convolutional neural networks",
        "contrast": "shallow models often rely more on manual feature engineering",
    },
    {
        "concept": "Supervised Learning",
        "definition": "learning from labeled input-output pairs",
        "example": "predicting house prices from historical examples",
        "contrast": "unsupervised learning has no target labels",
    },
    {
        "concept": "Unsupervised Learning",
        "definition": "learning structure from unlabeled data",
        "example": "customer segmentation with clustering",
        "contrast": "supervised learning optimizes against known labels",
    },
    {
        "concept": "Reinforcement Learning",
        "definition": "learning by interacting with an environment to maximize cumulative reward",
        "example": "game-playing agents trained with policy optimization",
        "contrast": "supervised learning learns from fixed labeled datasets",
    },
    {
        "concept": "Transformer",
        "definition": "a neural architecture based on self-attention for sequence modeling",
        "example": "BERT and GPT families",
        "contrast": "RNNs process tokens sequentially while transformers process tokens in parallel",
    },
    {
        "concept": "Self-Attention",
        "definition": "a mechanism that lets each token weigh other tokens in the same sequence",
        "example": "capturing long-range dependencies in text",
        "contrast": "fixed-window convolutions have limited direct receptive context",
    },
    {
        "concept": "Positional Encoding",
        "definition": "an approach to inject token order information into transformer inputs",
        "example": "sinusoidal vectors added to token embeddings",
        "contrast": "RNNs encode order through recurrence",
    },
    {
        "concept": "Tokenization",
        "definition": "splitting text into model-consumable units called tokens",
        "example": "byte-pair encoding used by many GPT models",
        "contrast": "word-level tokenization has larger vocabularies and more OOV issues",
    },
    {
        "concept": "Embedding",
        "definition": "a dense vector representation of tokens, words, or sentences",
        "example": "semantic search using sentence embeddings",
        "contrast": "one-hot vectors are sparse and do not encode similarity",
    },
    {
        "concept": "Fine-tuning",
        "definition": "adapting a pretrained model on a smaller task-specific dataset",
        "example": "tuning a language model for biomedical QA",
        "contrast": "pretraining uses broad corpora and high compute",
    },
    {
        "concept": "LoRA",
        "definition": "a parameter-efficient method that trains low-rank adapter matrices instead of all model weights",
        "example": "updating attention projections with rank-decomposed adapters",
        "contrast": "full fine-tuning updates every parameter",
    },
    {
        "concept": "PEFT",
        "definition": "parameter-efficient fine-tuning techniques for adapting large models with fewer trainable parameters",
        "example": "LoRA, prefix tuning, and adapter tuning",
        "contrast": "full fine-tuning is memory-intensive and costlier",
    },
    {
        "concept": "Prompt Engineering",
        "definition": "designing prompts to steer model behavior without changing weights",
        "example": "adding role instructions and output format constraints",
        "contrast": "fine-tuning modifies model parameters",
    },
    {
        "concept": "RAG",
        "definition": "retrieval-augmented generation that grounds answers using external documents",
        "example": "vector search over knowledge base before generation",
        "contrast": "pure parametric models rely only on stored training knowledge",
    },
    {
        "concept": "Hallucination",
        "definition": "a confident but incorrect model output",
        "example": "inventing citations that do not exist",
        "contrast": "grounded generation reduces unsupported claims",
    },
    {
        "concept": "Perplexity",
        "definition": "an intrinsic metric measuring how well a language model predicts tokens",
        "example": "lower perplexity often indicates better next-token modeling",
        "contrast": "task metrics like BLEU or ROUGE capture output quality differently",
    },
    {
        "concept": "BLEU",
        "definition": "an n-gram overlap metric commonly used for text generation evaluation",
        "example": "machine translation quality estimation",
        "contrast": "it may miss semantic correctness when wording differs",
    },
    {
        "concept": "ROUGE-L",
        "definition": "a metric based on longest common subsequence overlap between texts",
        "example": "summarization evaluation",
        "contrast": "exact match is stricter and less tolerant to paraphrase",
    },
    {
        "concept": "Exact Match",
        "definition": "the fraction of predictions that exactly match the reference after normalization",
        "example": "extractive QA benchmark scoring",
        "contrast": "BLEU and ROUGE give partial credit",
    },
    {
        "concept": "Overfitting",
        "definition": "when a model memorizes training data and generalizes poorly to unseen data",
        "example": "high train performance but low test performance",
        "contrast": "regularization and more data improve generalization",
    },
    {
        "concept": "Generalization",
        "definition": "a model's ability to perform well on unseen data",
        "example": "maintaining accuracy on held-out QA questions",
        "contrast": "memorization does not transfer reliably",
    },
    {
        "concept": "Learning Rate",
        "definition": "the step size used by the optimizer when updating model weights",
        "example": "2e-5 in transformer fine-tuning",
        "contrast": "too high causes instability and too low slows learning",
    },
    {
        "concept": "Batch Size",
        "definition": "the number of training samples processed before each optimizer step",
        "example": "small batches for memory-constrained local training",
        "contrast": "larger batches can improve throughput but require more memory",
    },
    {
        "concept": "Gradient Accumulation",
        "definition": "simulating a larger effective batch by accumulating gradients across mini-batches",
        "example": "accumulating 8 steps with batch size 2",
        "contrast": "direct large batches need more VRAM",
    },
    {
        "concept": "Causal Language Modeling",
        "definition": "training objective where each token predicts the next token",
        "example": "GPT-style autoregressive generation",
        "contrast": "masked language modeling predicts masked tokens",
    },
    {
        "concept": "Masked Language Modeling",
        "definition": "training objective that predicts intentionally masked tokens",
        "example": "BERT pretraining",
        "contrast": "causal models generate strictly left-to-right",
    },
    {
        "concept": "Instruction Tuning",
        "definition": "fine-tuning on instruction-response pairs to improve following user instructions",
        "example": "datasets with diverse tasks and explicit prompts",
        "contrast": "base pretraining is not optimized for conversational alignment",
    },
    {
        "concept": "Inference Optimization",
        "definition": "techniques to improve generation speed or memory usage at serving time",
        "example": "quantization and caching key-value states",
        "contrast": "training optimization focuses on model updates",
    },
    {
        "concept": "Top-k Sampling",
        "definition": "sampling from the k most probable next tokens",
        "example": "k=50 to avoid very low-probability tokens",
        "contrast": "greedy decoding always picks the single highest-probability token",
    },
    {
        "concept": "Top-p Sampling",
        "definition": "sampling from the smallest token set whose cumulative probability exceeds p",
        "example": "p=0.9 nucleus sampling",
        "contrast": "top-k uses a fixed number of tokens",
    },
    {
        "concept": "Temperature",
        "definition": "a scaling factor controlling randomness in token probabilities",
        "example": "lower temperature yields more deterministic outputs",
        "contrast": "higher temperature increases diversity but risk of errors",
    },
    {
        "concept": "Cross-Entropy Loss",
        "definition": "a loss measuring difference between predicted token distribution and true targets",
        "example": "standard objective for language model training",
        "contrast": "MSE is more common in regression tasks",
    },
    {
        "concept": "Evaluation Set",
        "definition": "a held-out subset used to estimate model performance during training",
        "example": "validation split used for model selection",
        "contrast": "test set should remain untouched until final reporting",
    },
]


QUESTION_TEMPLATES = {
    "definition": [
        "What is {concept} in AI/ML?",
        "Define {concept} in one sentence.",
        "How would you explain {concept}?",
        "What does {concept} mean?",
        "Give a short definition of {concept}.",
        "State the meaning of {concept}.",
        "In one technical sentence, what is {concept}?",
        "Explain {concept} briefly.",
        "How is {concept} defined?",
    ],
    "example": [
        "Give one example of {concept}.",
        "What is a practical example of {concept}?",
        "Show a real-world use of {concept}.",
        "Name one application of {concept}.",
        "Which example best illustrates {concept}?",
        "Provide a simple example of {concept}.",
        "Where do we see {concept} in practice?",
        "What is a common use case for {concept}?",
        "Give a concise example that shows {concept}.",
    ],
    "contrast": [
        "How does {concept} differ from a related approach?",
        "What is one key difference between {concept} and a similar method?",
        "Compare {concept} with a close alternative.",
        "What makes {concept} different from related ideas?",
        "In what way is {concept} not the same as a similar concept?",
        "Give one contrast between {concept} and a related technique.",
        "How can you distinguish {concept} from another ML approach?",
        "What is the main distinction for {concept}?",
        "Describe one difference between {concept} and a nearby alternative.",
    ],
    "practical": [
        "Why is {concept} useful in practice?",
        "When would {concept} matter in an ML workflow?",
        "How does {concept} help in real projects?",
        "What is the practical value of {concept}?",
        "Why do engineers use {concept}?",
        "How would {concept} be applied in an AI system?",
        "What role does {concept} play in practice?",
        "Why should an AI student learn {concept}?",
        "What is a practical reason to use {concept}?",
    ],
}


UPGRADED_QUESTION_TEMPLATES = {
    "technical_definition": [
        "In LLMs, what does {concept} mean during model behavior?",
        "For transformer-based NLP systems, define {concept} using technical terminology.",
        "In modern AI pipelines, what is the technical meaning of {concept}?",
        "In language-model training or inference, what does {concept} specifically refer to?",
    ],
    "technical_example": [
        "Give a technically precise definition of {concept} and include one concrete example.",
        "Explain {concept} and include a specific real-world ML or LLM example.",
        "For an exam-style answer, define {concept} and add one practical example.",
        "In production AI workflows, what is {concept} and what is one concrete use case?",
    ],
    "technical_contrast": [
        "Differentiate {concept} from a closely related method using precise terminology.",
        "In one concise technical answer, how does {concept} differ from nearby alternatives?",
        "In model design terms, what is the key distinction for {concept}?",
        "What technical trade-off best distinguishes {concept} from related approaches?",
    ],
}


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _build_answer(item: Dict[str, str], mode: str) -> str:
    concept = item["concept"]
    definition = item["definition"]
    example = item["example"]
    contrast = item["contrast"]

    if mode == "definition":
        return f"{concept} is {definition}."

    if mode == "example":
        return f"A practical example of {concept} is {example}."

    if mode == "contrast":
        return f"{concept} differs from related approaches because {contrast}."

    if mode == "technical_definition":
        return f"{concept} is {definition}."

    if mode == "technical_example":
        return f"{concept} is {definition}. For example, {example}."

    if mode == "technical_contrast":
        return (
            f"{concept} is {definition}. A key distinction is that {contrast}. "
            f"For example, {example}."
        )

    return f"{concept} is {definition}. For example, {example}."


def _build_curated_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    for item in CONCEPT_KB:
        concept = item["concept"]

        curated_questions = [
            f"What is {concept} in AI/ML?",
            f"Give one example of {concept}.",
            f"How is {concept} different from a related approach?",
            f"Why is {concept} useful in practice?",
        ]

        curated_answers = [
            _build_answer(item, "definition"),
            _build_answer(item, "example"),
            _build_answer(item, "contrast"),
            _build_answer(item, "practical"),
        ]

        for q, a in zip(curated_questions, curated_answers):
            examples.append({"question": q, "answer": a})

    return examples


def _build_upgraded_examples(target_size: int) -> List[Dict[str, str]]:
    upgraded: List[Dict[str, str]] = []
    seen_questions = set()

    modes = list(UPGRADED_QUESTION_TEMPLATES.keys())
    max_attempts = target_size * 20

    for index in range(max_attempts):
        if len(upgraded) >= target_size:
            break

        item = CONCEPT_KB[index % len(CONCEPT_KB)]
        mode = modes[(index // len(CONCEPT_KB)) % len(modes)]
        templates = UPGRADED_QUESTION_TEMPLATES[mode]
        template = templates[(index // (len(CONCEPT_KB) * len(modes))) % len(templates)]

        question = template.format(**item)
        answer = _build_answer(item, mode)

        question_key = _normalize_question(question)
        if question_key in seen_questions:
            continue

        seen_questions.add(question_key)
        upgraded.append({"question": question, "answer": answer})

    if len(upgraded) < target_size:
        raise RuntimeError(
            f"Could not generate enough upgraded samples: "
            f"requested={target_size}, generated={len(upgraded)}"
        )

    return upgraded


def _build_synthetic_examples(target_size: int) -> List[Dict[str, str]]:
    synthetic: List[Dict[str, str]] = []
    seen_questions = set()

    modes = list(QUESTION_TEMPLATES.keys())
    max_attempts = target_size * 10

    for index in range(max_attempts):
        if len(synthetic) >= target_size:
            break

        item = CONCEPT_KB[index % len(CONCEPT_KB)]
        mode = modes[(index // len(CONCEPT_KB)) % len(modes)]
        templates = QUESTION_TEMPLATES[mode]
        template = templates[(index // (len(CONCEPT_KB) * len(modes))) % len(templates)]

        question = template.format(**item)
        answer = _build_answer(item, mode)

        question_key = _normalize_question(question)
        if question_key in seen_questions:
            continue

        seen_questions.add(question_key)
        synthetic.append({"question": question, "answer": answer})

    if len(synthetic) < target_size:
        raise RuntimeError(
            f"Could not generate enough unique synthetic samples: "
            f"requested={target_size}, generated={len(synthetic)}"
        )

    return synthetic


def build_hybrid_dataset(total_samples: int = 650) -> List[Dict[str, str]]:
    """Build a small curated core plus deterministic template-based expansion."""
    curated = _build_curated_examples()

    curated_target = int(total_samples * 0.35)
    curated = curated[:curated_target]

    upgraded_target = min(UPGRADED_SAMPLE_COUNT, max(0, total_samples - len(curated)))
    upgraded = _build_upgraded_examples(target_size=upgraded_target)

    synthetic_target = total_samples - len(curated) - len(upgraded)
    synthetic = _build_synthetic_examples(target_size=synthetic_target)

    combined = curated + upgraded + synthetic
    RNG.shuffle(combined)
    return combined


def save_dataset_json(samples: List[Dict[str, str]], dataset_path: str) -> None:
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


def _save_split_files(dataset_dict: DatasetDict, processed_dir: str) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(processed_dir, f"{split}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(dataset_dict[split].to_list(), f, indent=2)


def _is_valid_dataset(samples: List[Dict[str, str]], expected_size: int) -> bool:
    if len(samples) != expected_size:
        return False

    for row in samples:
        if not isinstance(row, dict):
            return False
        if "question" not in row or "answer" not in row:
            return False
        if not isinstance(row["question"], str) or not isinstance(row["answer"], str):
            return False
        if not row["question"].strip() or not row["answer"].strip():
            return False

    return True


def create_dataset_splits(
    dataset_path: str,
    processed_dir: str,
    total_samples: int = 650,
    force_regenerate: bool = False,
) -> DatasetDict:
    """
    Create train/validation/test splits using 80/10/10 ratio.
    """
    regenerate = force_regenerate or not os.path.exists(dataset_path)

    if not regenerate:
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        regenerate = not _is_valid_dataset(samples, expected_size=total_samples)

    if regenerate:
        samples = build_hybrid_dataset(total_samples=total_samples)
        save_dataset_json(samples, dataset_path)

    dataset = Dataset.from_list(samples)

    train_test = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_valid = train_test["train"].train_test_split(test_size=0.111111, seed=SEED)

    dataset_dict = DatasetDict(
        {
            "train": train_valid["train"],
            "validation": train_valid["test"],
            "test": train_test["test"],
        }
    )

    _save_split_files(dataset_dict, processed_dir)
    return dataset_dict
