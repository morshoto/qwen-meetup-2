"""Canonical system instruction shared by Qwen inference and SFT records."""

FINAL_ANSWER_SYSTEM_PROMPT = 'Follow the requested output format exactly. Return only the final answer; do not explain.'

ANALYSIS_SYSTEM_PROMPT = (
    'Follow the requested format exactly. Give a concise, checkable summary of the observed pattern. '
    'Do not provide private chain-of-thought, code, or test answers.'
)


def system_prompt_for_kind(kind: str, *, analysis_mode: bool = False) -> str:
    """Select the legacy or explicitly experimental analysis instruction."""
    return (ANALYSIS_SYSTEM_PROMPT if analysis_mode and kind == 'analysis'
            else FINAL_ANSWER_SYSTEM_PROMPT)
