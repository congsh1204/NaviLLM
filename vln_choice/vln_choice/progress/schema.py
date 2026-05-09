from dataclasses import dataclass, field
from typing import Dict, List

from .normalize import tokenize, unique_normalized_terms


@dataclass
class Subgoal:
    text: str
    entities: List[str] = field(default_factory=list)
    action: str = "move"

    @property
    def match_terms(self) -> List[str]:
        terms = list(self.entities)
        terms.extend(tokenize(self.text))
        if self.action:
            terms.extend(tokenize(self.action))
        return unique_normalized_terms(terms)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entities": self.entities,
            "action": self.action,
        }


def coerce_subgoal(value) -> Subgoal:
    if isinstance(value, Subgoal):
        return value
    if isinstance(value, dict):
        return Subgoal(
            text=str(value.get("text", "")).strip(),
            entities=[str(x).strip() for x in value.get("entities", []) if str(x).strip()],
            action=str(value.get("action", "move")).strip() or "move",
        )
    return Subgoal(text=str(value).strip(), entities=[], action="move")


def coerce_subgoals(values) -> List[Subgoal]:
    return [sg for sg in (coerce_subgoal(value) for value in values or []) if sg.text]

