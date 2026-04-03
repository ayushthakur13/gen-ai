import re
import random
import string
from typing import Dict, List, Tuple

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "for", "to", "in", "on", "at", "by",
    "with", "as", "from", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "into", "about", "through", "across", "can", "could", "should", "would", "may", "might",
    "do", "does", "did", "done", "than", "then", "so", "such", "also", "very", "more", "most", "less", "least",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match_score(predictions: List[str], references: List[str]) -> float:
    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            matches += 1
    return matches / max(1, len(references))


def bleu_score(predictions: List[str], references: List[str]) -> float:
    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother))
    return sum(scores) / max(1, len(scores))


def rouge_l_score(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(score)
    return sum(scores) / max(1, len(scores))


def _keyword_tokens(text: str) -> List[str]:
    """
    Explicit keyword extraction rule:
    1) lowercase
    2) split tokens with regex
    3) remove stopwords
    """
    lowered = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", lowered)
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]


def keyword_overlap_score(predictions: List[str], references: List[str]) -> float:
    overlap_scores = []
    for pred, ref in zip(predictions, references):
        pred_set = set(_keyword_tokens(pred))
        ref_set = set(_keyword_tokens(ref))

        if not ref_set:
            overlap_scores.append(1.0 if not pred_set else 0.0)
            continue

        overlap = pred_set.intersection(ref_set)
        overlap_scores.append(len(overlap) / len(ref_set))

    return sum(overlap_scores) / max(1, len(overlap_scores))


def _manual_score_single(prediction: str, reference: str) -> Tuple[int, str]:
    """
    Manual-style scoring rubric (1-5) applied by evaluator for report consistency.
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    if pred_norm == ref_norm:
        return 5, "Exact technical match"

    pred_kw = set(_keyword_tokens(prediction))
    ref_kw = set(_keyword_tokens(reference))

    if not ref_kw:
        return 3, "Reference contains few keywords"

    recall = len(pred_kw.intersection(ref_kw)) / max(1, len(ref_kw))

    if recall >= 0.8:
        return 4, "Mostly correct with minor phrasing differences"
    if recall >= 0.55:
        return 3, "Partially correct but missing key details"
    if recall >= 0.25:
        return 2, "Weak answer with limited technical alignment"
    return 1, "Incorrect or largely unrelated answer"


def manual_human_evaluation(
    questions: List[str],
    predictions: List[str],
    references: List[str],
    sample_count: int = 8,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Evaluate 5-10 samples manually with a fixed rubric for assignment reporting.
    """
    sample_count = max(5, min(10, sample_count))
    sample_count = min(sample_count, len(questions))

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(questions)), sample_count))

    details = []
    numeric_scores = []

    for i in selected_indices:
        score, rationale = _manual_score_single(predictions[i], references[i])
        numeric_scores.append(score)
        details.append(
            {
                "question": questions[i],
                "prediction": predictions[i],
                "ground_truth": references[i],
                "score": score,
                "rationale": rationale,
            }
        )

    average_score = sum(numeric_scores) / max(1, len(numeric_scores))
    return {"average_score": average_score, "details": details}


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    return {
        "Exact Match": exact_match_score(predictions, references),
        "BLEU": bleu_score(predictions, references),
        "ROUGE-L": rouge_l_score(predictions, references),
        "Keyword Score": keyword_overlap_score(predictions, references),
    }
