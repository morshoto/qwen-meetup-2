"""Construct registered local components; no connection or execution occurs."""
from src.arc.http_backend import OpenAICompatibleBackend
from src.arc.lt_comparison import verify_manifest
from src.arc.sandbox import DockerSandbox


def prepare_components(frozen: dict, model: str) -> tuple[OpenAICompatibleBackend, DockerSandbox]:
    """Configuration consistency only; does not verify identities or authorize dispatch."""
    verify_manifest(frozen)
    if model not in ('14b', '32b'):
        raise ValueError('invalid model')
    config = frozen['config']
    backend = OpenAICompatibleBackend(model=config['model_names'][model], revision=config['models'][model],
        base_url=config['backend_url'], temperature=config['temperature'], max_tokens=config['max_tokens'],
        seed=config['seed'], timeout=config['timeout_seconds'], thinking=config['thinking'], guarded=True)
    sandbox = DockerSandbox(**config['sandbox'])
    return backend, sandbox


def check_guarded_metadata(frozen: dict, model: str, metadata: dict) -> dict:
    """Compare caller-supplied server metadata; this cannot authenticate the server."""
    verify_manifest(frozen)
    if model not in ('14b', '32b') or not isinstance(metadata, dict):
        raise ValueError('invalid server metadata')
    config = frozen['config']
    canary = metadata.get('thinking_canary')
    if (metadata.get('model') != config['model_names'][model] or
            metadata.get('revision') != config['models'][model] or
            type(metadata.get('token_limit')) is not int or
            metadata['token_limit'] != config['budget']['max_reserved_tokens'] or
            type(metadata.get('reserved_tokens')) is not int or metadata['reserved_tokens'] != 0 or
            type(metadata.get('max_prompt_tokens')) is not int or metadata['max_prompt_tokens'] <= 0 or
            type(metadata.get('max_new_tokens')) is not int or metadata['max_new_tokens'] != config['max_tokens'] or
            metadata.get('max_request_seconds') != config['timeout_seconds'] or
            metadata.get('contract') != 'same_process_token_ids_v1' or
            metadata.get('hard_deadline') != 'process_exit_v1' or
            metadata.get('adapter_sha256') != 'none' or metadata.get('thinking_mode') != config['thinking'] or
            not isinstance(metadata.get('thinking_modes'), list) or config['thinking'] not in metadata['thinking_modes'] or
            not isinstance(canary, dict) or canary.get('supported') is not True or
            canary.get('different_token_ids') is not True):
        raise ValueError('guarded server metadata differs from registration')
    return {'status': 'metadata_consistent_unverified', 'model': model,
            'comparison_sha256': frozen['sha256'], 'execution_ready': False,
            'server_identity_independently_verified': False}


def probe_guarded_server(frozen: dict, model: str) -> dict:
    """Read loopback metadata only; no generation request or trust decision is made."""
    import hashlib
    import urllib.error
    import urllib.request
    from src.arc.http_backend import _NoRedirect
    from src.arc.lt_attestation import _object

    verify_manifest(frozen)
    if model not in ('14b', '32b'):
        raise ValueError('invalid model')
    url = frozen['config']['backend_url'] + '/token-budget'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(urllib.request.Request(url, method='GET'), timeout=5) as response:
            if response.status != 200:
                raise ValueError('guarded server metadata unavailable')
            raw = response.read(65537)
        if not raw or len(raw) > 65536:
            raise ValueError('guarded server metadata size invalid')
        report = check_guarded_metadata(frozen, model, _object(raw))
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError('guarded server metadata unavailable') from exc
    return {**report, 'metadata_sha256': hashlib.sha256(raw).hexdigest()}
