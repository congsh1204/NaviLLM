import json
import re
from typing import Iterable, Optional


def extract_json(text: str):
    """Extract the first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")
    return json.loads(match.group(0))


def parse_final_choice(text: str, valid_choices: Iterable[str]) -> Optional[str]:
    valid = set(valid_choices)
    try:
        obj = extract_json(text)
        choice = str(obj.get("final_choice", "")).strip()
        return choice if choice in valid else None
    except Exception:
        for choice in valid:
            if re.search(r"\b{}\b".format(re.escape(choice)), text):
                return choice
    return None

