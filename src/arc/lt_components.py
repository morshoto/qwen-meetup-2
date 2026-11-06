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
