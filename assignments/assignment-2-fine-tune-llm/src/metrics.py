import re
import string
from typing import Dict, List

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


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    return {
        "BLEU": bleu_score(predictions, references),
        "ROUGE-L": rouge_l_score(predictions, references),
        "Keyword Score": keyword_overlap_score(predictions, references),
    }
