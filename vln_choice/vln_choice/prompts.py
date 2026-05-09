import json
from typing import Dict, Iterable, List


EGAC_STATES = ("defer", "prepare", "commit", "stop")


def build_choice_prompt(instruction: str, option_labels: Iterable[str], history: List[str] = None, with_cot: bool = False) -> str:
    history = history or []
    labels = ", ".join(option_labels)
    history_text = "\n".join("- {}".format(item) for item in history) if history else "None"
    if with_cot:
        output_spec = (
            'Return valid JSON only with keys "candidate_analysis" and "final_choice". '
            'final_choice must be one of: {}. Keep each evidence sentence short.'
        ).format(labels)
    else:
        output_spec = 'Return valid JSON only: {{"final_choice": "<one of: {}>"}}.'.format(labels)

    return (
        "You are an embodied navigation agent.\n\n"
        "The image is a composite: the top part is the current panorama, and the bottom part shows candidate views.\n"
        "Choose the best next action according to the instruction and recent history.\n\n"
        "Instruction:\n{}\n\n"
        "Recent history:\n{}\n\n"
        "{}"
    ).format(instruction, history_text, output_spec)


def build_egac_prompt(instruction: str, option_labels: Iterable[str], history: List[str] = None) -> str:
    """Prompt that asks for Evidence-Gated Action Commitment (EGAC) structured output.

    The model must reason about whether visual evidence currently supports committing to
    the next target action, before picking a candidate. See ``egac_response`` for the
    matching JSON schema.
    """
    history = history or []
    labels = ", ".join(option_labels)
    states = ", ".join(EGAC_STATES)
    history_text = "\n".join("- {}".format(item) for item in history) if history else "None"
    output_spec = (
        'Return valid JSON only with the keys: '
        '"target_action", "required_evidence", "observed_evidence", '
        '"commitment_state", "next_action", "final_choice". '
        'commitment_state must be one of: {states}. '
        'final_choice must be one of: {labels}. '
        'required_evidence and observed_evidence are short noun phrases (lists). '
        'Use "defer" when required evidence is not yet visible, "prepare" when partially visible, '
        '"commit" only when evidence is sufficient to execute the target action, '
        'and "stop" only when the target has been reached.'
    ).format(states=states, labels=labels)

    return (
        "You are an embodied navigation agent that decides when to commit to an action based on visual evidence.\n\n"
        "The image is a composite: the top part is the current panorama, and the bottom part shows candidate views.\n"
        "Before choosing a candidate, judge whether the evidence required by the next target action is currently observed.\n"
        "If evidence is insufficient, prefer approaching/preparing rather than executing the action prematurely.\n\n"
        "Instruction:\n{instr}\n\n"
        "Recent history:\n{hist}\n\n"
        "{spec}"
    ).format(instr=instruction, hist=history_text, spec=output_spec)


def final_choice_response(final_choice: str) -> str:
    return json.dumps({"final_choice": final_choice}, ensure_ascii=False)


def cot_response(candidate_analysis: Dict[str, Dict[str, object]], final_choice: str) -> str:
    return json.dumps(
        {"candidate_analysis": candidate_analysis, "final_choice": final_choice},
        ensure_ascii=False,
    )


def egac_response(
    target_action: str,
    required_evidence: List[str],
    observed_evidence: List[str],
    commitment_state: str,
    next_action: str,
    final_choice: str,
) -> str:
    """Serialize a single EGAC training target.

    ``commitment_state`` must be one of :data:`EGAC_STATES`. Field order is fixed so
    JSONL diffs and tokenization stay deterministic across regeneration.
    """
    if commitment_state not in EGAC_STATES:
        raise ValueError("commitment_state {!r} not in {}".format(commitment_state, EGAC_STATES))
    return json.dumps(
        {
            "target_action": target_action,
            "required_evidence": list(required_evidence),
            "observed_evidence": list(observed_evidence),
            "commitment_state": commitment_state,
            "next_action": next_action,
            "final_choice": final_choice,
        },
        ensure_ascii=False,
    )

