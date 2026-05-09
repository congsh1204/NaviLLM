import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import List, Optional

from .normalize import normalize_token, tokenize
from .schema import Subgoal


def _norm_keyword_set(*keywords: str):
    return {normalize_token(k) for k in keywords}


# enter / turn / stop first; vertical motion handled separately so "stairs" alone is not mislabeled.
_PRIMARY_ACTION_KEYWORDS = {
    "enter": _norm_keyword_set("enter", "entrance", "door", "doorway", "through", "into"),
    "turn": _norm_keyword_set("turn", "left", "right", "around"),
    "stop": _norm_keyword_set("stop", "wait"),
}

# Directional only — do not put stairs/stair here (ambiguous without up/down).
_ASCEND_MARKERS = _norm_keyword_set("up", "ascend", "upstairs", "upward", "climb")
_DESCEND_MARKERS = _norm_keyword_set("down", "descend", "downstairs", "downward")


def infer_action(text: str) -> str:
    tokens = set(tokenize(text, keep_stopwords=True))
    # Prefer vertical motion before generic "stop"/other primaries so clauses like
    # "walk down ... and stop on the landing" label as descend (down), not stop.
    ascend = bool(tokens & _ASCEND_MARKERS)
    descend = bool(tokens & _DESCEND_MARKERS)
    if ascend and descend:
        return "move"
    if ascend:
        return "ascend"
    if descend:
        return "descend"

    for action, kw_set in _PRIMARY_ACTION_KEYWORDS.items():
        if tokens & kw_set:
            return action
    return "move"


def extract_entities(text: str) -> List[str]:
    tokens = tokenize(text)
    action_words = {"move", "enter", "turn", "stop", "descend", "ascend", "left", "right", "forward", "straight"}
    out = []
    seen = set()
    for token in tokens:
        if token in action_words or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:8]


def heuristic_split(instruction: str) -> List[dict]:
    chunks = []
    parts = re.split(r"(?:[.;]+|\bthen\b|\band then\b|\bafter that\b)", instruction, flags=re.IGNORECASE)
    for part in parts:
        text = " ".join(part.strip().split())
        if not text:
            continue
        chunks.append(Subgoal(text=text, entities=extract_entities(text), action=infer_action(text)).to_dict())
    if not chunks and instruction.strip():
        text = " ".join(instruction.strip().split())
        chunks.append(Subgoal(text=text, entities=extract_entities(text), action=infer_action(text)).to_dict())
    return chunks


def _extract_json_array(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if isinstance(obj, dict):
        obj = obj.get("subgoals", obj.get("new_instructions", []))
    return obj


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from model output; tolerate fences and leading/trailing text."""
    t = _strip_markdown_fences(text)
    try:
        obj = json.loads(t)
    except Exception:
        start = t.find("{")
        if start < 0:
            raise ValueError("no JSON object in model output") from None
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object, got {}".format(type(obj).__name__))
    return obj


def build_subgoal_split_user_prompt(instruction: str) -> str:
    """Compact user prompt for FourZApiSplitter (token-efficient, strict JSON)."""
    # Do not use .format() on the whole block: literal JSON braces like {"text":...} would be
    # parsed as format fields if chained with "Instruction: {}".format(...).
    body = (
        "Return one JSON array only—no markdown, fences, or commentary. "
        "Split the indoor navigation instruction into ordered subgoals (execution order). "
        'Each element: {"text": string, "entities": string[], "action": string}. '
        "text: imperative clause for that step. "
        "entities: 1–6 short landmark/room/object phrases for grounding (nouns); skip filler words. "
        "action: exactly one of move, enter, turn, stop, ascend, descend. "
        "Do not assume the first subgoal already satisfies the main goal of the instruction. "
        "Avoid inventing search-like actions (find/look for/search) unless the instruction explicitly asks to search/find/look for something. "
        "Do not invert directional meaning: if instruction says down/downstairs/descend, do not output ascend; "
        "if it says up/upstairs/ascend, do not output descend. "
        "Keep prerequisite navigation before vertical movement (e.g., approach stairs before going down/up). "
        "Language understanding only—do not invent path IDs, step indices, or distances.\n\n"
    )
    return body + "Instruction: {}\n".format(instruction.strip())


def build_step_alignment_user_prompt(instruction: str, subgoals: List[dict], step_evidence: List[dict]) -> str:
    """Prompt for LLM step->subgoal alignment."""
    payload = {
        "instruction": instruction,
        "subgoals": subgoals,
        "steps": step_evidence,
        "requirements": {
            "output_only_json": True,
            "step_subgoal_index_len_equals_steps": True,
            "index_is_0_based": True,
            "allow_terminal_stop_alignment": True,
            "prefer_monotonic_non_decreasing": True,
        },
    }
    return (
        "Return one JSON object only, no markdown or commentary.\n"
        "Task: assign each path step to one subgoal index. "
        "Each step includes landmarks visible along the expert path at that path_pos—use them to decide "
        "whether a subgoal's target is already present/relevant at that step. "
        "If a subgoal requires object O but O's landmarks only appear in later steps, earlier steps should align "
        "to a subgoal that describes approaching or reaching O (not performing O's main action there). "
        "Do not assume the first step already satisfies the main instruction action.\n"
        'Output schema: {"step_subgoal_index":[int,...], "confidence":"low|medium|high"}\n'
        "Rules:\n"
        "- Each index must be within [0, len(subgoals)-1].\n"
        "- step_subgoal_index length must equal steps length.\n"
        "- Respect action direction and step landmarks; avoid random oscillation.\n"
        "- Prefer non-decreasing progression unless strong evidence says otherwise.\n\n"
        "Input JSON:\n{}\n".format(json.dumps(payload, ensure_ascii=False))
    )


def build_joint_split_and_alignment_user_prompt(instruction: str, step_evidence: List[dict]) -> str:
    """Single-call prompt: split subgoals then align each step to subgoal index."""
    payload = {
        "instruction": instruction,
        "steps": step_evidence,
        "requirements": {
            "output_only_json": True,
            "split_subgoals_first": True,
            "then_assign_steps": True,
            "index_is_0_based": True,
            "prefer_monotonic_non_decreasing": True,
            "actions_allowed": ["move", "enter", "turn", "stop", "ascend", "descend"],
        },
    }
    body = (
        "Return one JSON object only, no markdown.\n"
        "Given the original navigation instruction and the landmark sequence along the expert path "
        "(each entry in steps has path_pos and landmarks at that position), split the instruction into "
        "fine-grained ordered subgoals, then assign each step to exactly one subgoal index.\n"
        "Do not assume the first step already satisfies the main action of the instruction. "
        "For each subgoal, check whether its required target object or place is supported by the landmarks at the "
        "path positions you assign to it—targets should appear or become plausible as those steps progress. "
        "If an action requires object O but O does not appear in early steps' landmarks and appears only later, "
        "use separate subgoals: earlier phases should read as approaching or reaching O (e.g. move toward / navigate "
        "to O); label steps where O is absent only as that approach phase, not as completing O's main action. "
        "Avoid adding unsupported search actions such as find/search/look for O unless the instruction explicitly "
        "says to search, find, or look for something.\n"
        "The downstream pipeline derives chunk_view (path_pos ranges per subgoal) from your step_subgoal_index; "
        "choose indices so those ranges align with landmark-supported phases.\n"
        "Do two tasks in order:\n"
        "1) Split into subgoals (concise text, entities, action types).\n"
        "2) Assign each step to a subgoal index.\n"
        'Output schema: {"subgoals":[{"text":string,"entities":string[],"action":string},...], '
        '"step_subgoal_index":[int,...], "confidence":"low|medium|high"}\n'
        "Rules:\n"
        "- Keep directional meaning (down != up).\n"
        "- step_subgoal_index length must equal steps length.\n"
        "- Each index in [0, len(subgoals)-1].\n"
        "- Prefer non-decreasing progression unless strong evidence says otherwise.\n\n"
    )
    return body + "Input JSON:\n{}\n".format(json.dumps(payload, ensure_ascii=False))


def _fix_directional_action_consistency(subgoals: List[dict], instruction: str) -> List[dict]:
    """Correct obvious up/down polarity flips from remote splitter output."""
    instr_tokens = set(tokenize(instruction, keep_stopwords=True))
    instr_has_ascend = bool(instr_tokens & _ASCEND_MARKERS)
    instr_has_descend = bool(instr_tokens & _DESCEND_MARKERS)

    fixed = []
    for sg in subgoals:
        text = str(sg.get("text", ""))
        action = str(sg.get("action", "")).strip().lower()
        sg_tokens = set(tokenize(text, keep_stopwords=True))
        sg_has_ascend = bool(sg_tokens & _ASCEND_MARKERS)
        sg_has_descend = bool(sg_tokens & _DESCEND_MARKERS)

        if action == "ascend" and (sg_has_descend or (instr_has_descend and not instr_has_ascend)):
            sg = dict(sg)
            sg["action"] = "descend"
        elif action == "descend" and (sg_has_ascend or (instr_has_ascend and not instr_has_descend)):
            sg = dict(sg)
            sg["action"] = "ascend"
        fixed.append(sg)
    return fixed


def _subgoals_from_parsed(parsed, instruction: str) -> List[dict]:
    subgoals = []
    for item in parsed:
        if isinstance(item, str):
            text = item.strip()
            subgoals.append(Subgoal(text=text, entities=extract_entities(text), action=infer_action(text)).to_dict())
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            entities = [str(x).strip() for x in item.get("entities", []) if str(x).strip()]
            subgoals.append(
                Subgoal(
                    text=text,
                    entities=entities or extract_entities(text),
                    action=str(item.get("action", infer_action(text))).strip() or infer_action(text),
                ).to_dict()
            )
    out = subgoals or heuristic_split(instruction)
    return _fix_directional_action_consistency(out, instruction)


def _is_timeout_like(exc: BaseException) -> bool:
    """Detect socket/read timeouts for selective retries."""
    if isinstance(exc, (socket.timeout, TimeoutError, BrokenPipeError, ConnectionResetError)):
        return True
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    if isinstance(exc, urllib.error.URLError) and exc.reason is not None:
        return _is_timeout_like(exc.reason)
    return False


class FourZApiSplitter:
    """Remote LLM via HTTP (4Z API / OpenAI-compatible chat completions).

    Keys and model ids: https://4zapi.com — use the console OpenAI-compatible API Base URL
    (typically ends with /v1). Requests go to {base}/chat/completions with Bearer auth.

    Env (recommended):
      FOURZ_API_BASE or API_4Z_BASE   console base URL, e.g. https://.../v1
      FOURZ_API_KEY or API_4Z_KEY     API key from console
      FOURZ_MODEL or API_4Z_MODEL     model id as listed on the platform (build_progress_labels defaults --model_name_or_path to gpt-5.4-mini if unset)

    Optional:
      FOURZ_AUTH_HEADER / API_4Z_AUTH_HEADER  full Authorization header value
      FOURZ_TIMEOUT_SEC / API_4Z_TIMEOUT_SEC  request timeout seconds (default 120 if unset; use env to increase)
      FOURZ_CHAT_RETRIES / API_4Z_CHAT_RETRIES  transient read-timeout retries for _chat (default 3)
    """

    def __init__(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
        max_new_tokens: int = 512,
        timeout_sec: Optional[float] = None,
        chat_retries: Optional[int] = None,
    ):
        self.api_base = (
            api_base
            or os.environ.get("FOURZ_API_BASE")
            or os.environ.get("API_4Z_BASE")
            or ""
        ).strip().rstrip("/")
        self.api_key = (
            api_key or os.environ.get("FOURZ_API_KEY") or os.environ.get("API_4Z_KEY") or ""
        ).strip()
        self.model = (
            model or os.environ.get("FOURZ_MODEL") or os.environ.get("API_4Z_MODEL") or ""
        ).strip()
        self.max_new_tokens = max_new_tokens
        env_timeout = os.environ.get("FOURZ_TIMEOUT_SEC") or os.environ.get("API_4Z_TIMEOUT_SEC")
        self.timeout_sec = timeout_sec
        if self.timeout_sec is None and env_timeout:
            try:
                self.timeout_sec = float(env_timeout)
            except ValueError:
                self.timeout_sec = 120.0
        if self.timeout_sec is None:
            self.timeout_sec = 120.0

        env_retries = os.environ.get("FOURZ_CHAT_RETRIES") or os.environ.get("API_4Z_CHAT_RETRIES")
        self.chat_retries = chat_retries
        if self.chat_retries is None and env_retries:
            try:
                self.chat_retries = max(1, int(env_retries))
            except ValueError:
                self.chat_retries = 3
        if self.chat_retries is None:
            self.chat_retries = 3

        if not self.api_base:
            raise ValueError(
                "api_4z requires FOURZ_API_BASE or --api_base "
                "(OpenAI-compatible URL ending in /v1 from https://4zapi.com or your provider)."
            )
        if not self.model:
            raise ValueError(
                "api_4z requires FOURZ_MODEL, --api_model, or --model_name_or_path as remote model id."
            )

    def _chat(self, user_content: str) -> str:
        url = "{}/chat/completions".format(self.api_base)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        auth_full = os.environ.get("FOURZ_AUTH_HEADER") or os.environ.get("API_4Z_AUTH_HEADER")
        if auth_full:
            req.add_header("Authorization", auth_full.strip())
        elif self.api_key:
            req.add_header("Authorization", "Bearer {}".format(self.api_key))

        last_exc: Optional[BaseException] = None
        raw = None
        for attempt in range(self.chat_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError("4ZAPI HTTP {}: {}".format(exc.code, body[:2000])) from exc
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.chat_retries and _is_timeout_like(exc):
                    time.sleep(min(8.0, 2.0**attempt))
                    continue
                raise
        if raw is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("4ZAPI: empty response")

        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError("4ZAPI empty choices: {}".format(raw[:2000]))
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text", "")
        return str(content).strip()

    def split(self, instruction: str) -> List[dict]:
        prompt = build_subgoal_split_user_prompt(instruction)
        try:
            output = self._chat(prompt)
            parsed = _extract_json_array(output)
            return _subgoals_from_parsed(parsed, instruction)
        except Exception:
            return heuristic_split(instruction)

    def align_steps(self, instruction: str, subgoals: List[dict], step_evidence: List[dict]):
        """LLM step->subgoal assignment, returns dict with indices and confidence."""
        prompt = build_step_alignment_user_prompt(instruction, subgoals, step_evidence)
        try:
            output = self._chat(prompt)
            obj = _extract_json_object(output)
            idxs = obj.get("step_subgoal_index")
            if not isinstance(idxs, list):
                raise ValueError("missing step_subgoal_index")
            out = []
            m = max(1, len(subgoals))
            for x in idxs:
                try:
                    k = int(x)
                except Exception:
                    k = 0
                out.append(max(0, min(m - 1, k)))
            if len(out) < len(step_evidence):
                out.extend([out[-1] if out else 0] * (len(step_evidence) - len(out)))
            if len(out) > len(step_evidence):
                out = out[: len(step_evidence)]
            return {
                "step_subgoal_index": out,
                "confidence": str(obj.get("confidence", "medium")),
                "source": "llm_step_alignment",
            }
        except Exception as exc:
            # fallback: assign all to first subgoal
            return {
                "step_subgoal_index": [0] * len(step_evidence),
                "confidence": "low",
                "source": "llm_step_alignment_fallback",
                "llm_error": str(exc)[:2000],
            }

    def split_and_align(self, instruction: str, step_evidence: List[dict]):
        """Single API call: split subgoals and align step indices."""
        prompt = build_joint_split_and_alignment_user_prompt(instruction, step_evidence)
        try:
            output = self._chat(prompt)
            obj = _extract_json_object(output)
            parsed_subgoals = obj.get("subgoals", [])
            subgoals = _subgoals_from_parsed(parsed_subgoals, instruction)
            idxs = obj.get("step_subgoal_index", [])
            m = max(1, len(subgoals))
            clean = []
            for x in idxs:
                try:
                    k = int(x)
                except Exception:
                    k = 0
                clean.append(max(0, min(m - 1, k)))
            if len(clean) < len(step_evidence):
                clean.extend([clean[-1] if clean else 0] * (len(step_evidence) - len(clean)))
            if len(clean) > len(step_evidence):
                clean = clean[: len(step_evidence)]
            return {
                "subgoals": subgoals,
                "step_subgoal_index": clean,
                "confidence": str(obj.get("confidence", "medium")),
                "source": "llm_split_and_alignment",
            }
        except Exception as exc:
            fallback_subgoals = heuristic_split(instruction)
            return {
                "subgoals": fallback_subgoals,
                "step_subgoal_index": [0] * len(step_evidence),
                "confidence": "low",
                "source": "llm_split_and_alignment_fallback",
                "llm_error": str(exc)[:2000],
            }


def get_splitter(
    name: str,
    model_name_or_path: Optional[str] = None,
    max_new_tokens: int = 512,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    chat_retries: Optional[int] = None,
):
    if name == "heuristic":
        return None
    if name == "api_4z":
        model = (
            api_model
            or model_name_or_path
            or os.environ.get("FOURZ_MODEL")
            or os.environ.get("API_4Z_MODEL")
        )
        return FourZApiSplitter(
            api_base=api_base,
            api_key=api_key,
            model=model,
            max_new_tokens=max_new_tokens,
            timeout_sec=timeout_sec,
            chat_retries=chat_retries,
        )
    raise ValueError("Unknown splitter: {}".format(name))

