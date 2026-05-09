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

