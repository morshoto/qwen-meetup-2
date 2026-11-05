"""Compare saved and replayed prompt scans without attesting their origin."""
import re

_STATUSES = {'exact':'bounded_exact_prompt_review_only', 'hash':'bounded_historical_prompt_hash_review_only',
             'text':'bounded_historical_task_text_review_only'}


def reconcile_prompt_reviews(proposal, original_split, split_sha256, reviews, replay):
    try:
        development = proposal['partitions']['development']
        if (proposal.get('status') != 'bounded_exposure_proposed_unverified' or
                proposal.get('execution_ready') is not False or not isinstance(development, list) or
                not development or len(set(development)) != len(development) or
                development != original_split['partitions']['development'] or
                not isinstance(split_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}',split_sha256) or
                set(reviews) != set(_STATUSES) or set(replay) != set(_STATUSES)):
            raise ValueError('proposal or review set differs')
        inventories = set()
        for name,status in _STATUSES.items():
            saved, current = reviews[name], replay[name]
            if (saved.get('status') != status or saved.get('independence_established') is not False or
                    saved.get('dataset_sha256') != proposal['task_source_sha256'] or
                    saved.get('split_sha256') != split_sha256 or
                    type(saved.get('file_count')) is not int or saved['file_count'] < 1 or
                    not isinstance(saved.get('inventory_sha256'),str) or
                    not re.fullmatch(r'[0-9a-f]{64}',saved['inventory_sha256']) or
                    any(saved.get(k) != current.get(k) for k in ('file_count','inventory_sha256','matches'))):
                raise ValueError('saved review or replay differs')
            if saved['matches'] != {}:
                raise ValueError('possible development exposure detected')
            inventories.add((saved['file_count'],saved['inventory_sha256']))
        if len(inventories) != 1:
            raise ValueError('prompt review inventories differ')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid prompt review evidence') from exc
    return {'status':'prompt_reviews_replayed_unverified', 'execution_ready':False,
            'independent_review_completed':False, 'development_count':len(development),
            'artifact_file_count':next(iter(inventories))[0], 'artifact_inventory_sha256':next(iter(inventories))[1]}
