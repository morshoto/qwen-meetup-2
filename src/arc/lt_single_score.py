"""Score one registered model with the existing Phase A scoring rules."""
from src.arc.lt_scoring import score_comparison


def score_single(task_ids, rows, verifier):
    if any(not isinstance(row, dict) or row.get('model') != 'single' for row in rows):
        raise ValueError('invalid model')
    result = score_comparison(task_ids, [{**row, 'model': '14b'} for row in rows], verifier)
    return {'rows': [{**row, 'model': 'single'} for row in result['rows'] if row['model'] == '14b'],
            'models': {'single': result['models']['14b']}}
