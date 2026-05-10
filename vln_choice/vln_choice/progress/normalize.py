import re
from typing import Iterable, List


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "go",
    "into",
    "in",
    "is",
    "of",
    "on",
    "past",
    "the",
    "then",
    "there",
    "to",
    "toward",
    "towards",
    "turn",
    "walk",
    "with",
}

_ALIASES = {
    "stair": "stairs",
    "staircase": "stairs",
    "step": "stairs",
    "steps": "stairs",
    "couch": "sofa",
    "settee": "sofa",
    "picture": "art",
    "painting": "art",
    "photo": "art",
    "entrance": "door",
    "doorway": "door",
    "corridor": "hallway",
    "hall": "hallway",
    "lavatory": "bathroom",
    "restroom": "bathroom",
    "tv": "television",
}


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9]+", "", token)
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss") and token not in {"stairs"}:
        token = token[:-1]
    return _ALIASES.get(token, token)


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z0-9]+", text.lower()):
        token = normalize_token(raw)
        if not token:
            continue
        if not keep_stopwords and token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text, keep_stopwords=False))


def unique_normalized_terms(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        for token in tokenize(value):
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


# Tokens that show up as ``entities`` from the LLM splitter but are not actual physical
# objects/places, so they never appear as t2t landmark tokens. Filtering them on the
# **entity side** of an overlap check stops them from forcing artificial false negatives
# (currently ~70% of subgoals were flagged "no evidence hit" largely because of these).
# Audit source: token-level frequency analysis on r2r_progress_labels.jsonl, May 2026.
_NON_ENTITY_TOKENS = {
    # directions
    "left", "right", "front", "back", "ahead", "behind", "forward", "backward",
    "north", "south", "east", "west",
    # generic spatial nouns
    "area", "place", "side", "way", "direction", "spot", "position", "location",
    # modal / state verbs the splitter sometimes mistakes for objects
    "wait", "open", "close", "stop", "enter", "exit",
    # vague determiners that survive tokenize
    "thing", "stuff",
}


def entity_tokens(text: str) -> List[str]:
    """Same as ``tokenize`` but additionally drops directional / generic-spatial / modal
    tokens that the splitter often mis-extracts as entities. Use this on the **entity**
    side of an entity↔landmark overlap; landmarks should still go through plain ``tokenize``.
    """
    return [t for t in tokenize(text) if t and t not in _NON_ENTITY_TOKENS]

