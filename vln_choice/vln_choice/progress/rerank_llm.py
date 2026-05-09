from typing import Dict, List


class RerankerNotConfigured(RuntimeError):
    pass


def rerank_candidates_with_llm(*args, **kwargs) -> Dict:
    """Placeholder for v2 LLM reranking.

    v1 intentionally keeps chunk_view generation fully deterministic via DP.
    Wire a Qwen judge here after v1 labels and confidence filtering are validated.
    """
    raise RerankerNotConfigured("LLM reranking is reserved for v2 and is not configured in this v1 implementation.")


def passthrough_best(candidates: List[Dict]) -> Dict:
    if not candidates:
        return {"best_index": None, "confidence": "low", "reason": "No DP candidates."}
    return {"best_index": 0, "confidence": "high", "reason": "Selected highest-scoring DP candidate."}

