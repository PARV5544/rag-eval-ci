import re
from dataclasses import dataclass
from typing import Dict, List, Set

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "this", "these", "those", "or", "not",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


@dataclass
class SampleResult:
    sample_id: str
    faithfulness: float
    answer_relevance: float
    hallucinated: bool
    unsupported_sentences: List[str]


def score_sample(
    sample_id: str,
    question: str,
    answer: str,
    contexts: List[str],
    sentence_support_threshold: float = 0.5,
    faithfulness_cutoff: float = 0.7,
) -> SampleResult:
    """Evaluate how well an answer's claims are grounded in retrieved contexts."""
    context_tokens: Set[str] = set()
    for ctx in contexts:
        context_tokens.update(_tokenize(ctx))

    sentences = _split_sentences(answer)
    if not sentences:
        return SampleResult(
            sample_id=sample_id,
            faithfulness=0.0,
            answer_relevance=0.0,
            hallucinated=True,
            unsupported_sentences=[],
        )

    supported_count = 0
    unsupported: List[str] = []
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            supported_count += 1
            continue
        overlap = sum(1 for tok in sent_tokens if tok in context_tokens)
        ratio = overlap / len(sent_tokens)
        if ratio >= sentence_support_threshold:
            supported_count += 1
        else:
            unsupported.append(sent)

    faithfulness = supported_count / len(sentences)

    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer))
    if q_tokens and a_tokens:
        answer_relevance = len(q_tokens & a_tokens) / len(q_tokens)
    else:
        answer_relevance = 0.0

    return SampleResult(
        sample_id=sample_id,
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(answer_relevance, 4),
        hallucinated=faithfulness < faithfulness_cutoff,
        unsupported_sentences=unsupported,
    )


def aggregate_results(results: List[SampleResult]) -> Dict[str, float]:
    """Compute aggregate faithfulness, relevance, and hallucination rate."""
    if not results:
        return {
            "total_samples": 0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevance": 0.0,
            "hallucination_rate": 1.0,
        }

    total = len(results)
    avg_faith = sum(r.faithfulness for r in results) / total
    avg_rel = sum(r.answer_relevance for r in results) / total
    hallucinated_count = sum(1 for r in results if r.hallucinated)
    return {
        "total_samples": total,
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevance": round(avg_rel, 4),
        "hallucination_rate": round(hallucinated_count / total, 4),
    }
