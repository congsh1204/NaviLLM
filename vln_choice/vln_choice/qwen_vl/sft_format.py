"""Single source for chat `messages` layout used by LoRA SFT — keep in sync with inference user turns."""


def format_example_for_sft(example: dict) -> dict:
    """Map one JSONL row (image, prompt, response) to TRL message dict."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]},
        ]
    }
