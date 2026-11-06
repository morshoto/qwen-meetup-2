"""Reconstruct historical task blocks without claiming exposure."""
from src.arc.agent import _code_task_text
from src.arc.rsi_historical_prompt_hash import historical_prompt_hashes


def historical_task_texts(tasks):
    # Reuse the same task and grid validation as the historical hash review.
    historical_prompt_hashes(tasks)
    return {task.id: _code_task_text(task) for task in tasks}
