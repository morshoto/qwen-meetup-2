import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
EXTERNAL_DIR = DATA_DIR / "external"

COMPETITION = os.getenv("COMPETITION", "")
SEED = int(os.getenv("SEED", "42"))


def ensure_data_dirs() -> None:
    """Create standard data directories if they do not exist."""
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or os.getenv("KAGGLE_URL_BASE") is not None


def resolve_raw_dir() -> Path:
    if os.getenv("RAW_DIR"):
        return Path(os.getenv("RAW_DIR")).expanduser()

    if is_kaggle() and COMPETITION:
        return Path("/kaggle/input") / COMPETITION

    data_dir = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
    return data_dir / "raw"


RAW_DIR = resolve_raw_dir()

# The experiment configuration is intentionally separate from the template's path helpers.
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
import math
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seed: int
    conditions: list[str]
    generations: int
    evaluation_generations: list[int]
    evaluation_split: str


@dataclass(frozen=True)
class ModelSettings:
    id: str
    revision: str
    backend: str
    temperature: float
    max_tokens: int
    base_url: str = 'http://127.0.0.1:8080/v1'
    timeout_seconds: float = 120.0
    thinking: str = 'server_default'
    allow_remote: bool = False
    quantization: str = 'unknown'
    adapter_sha256: str = 'none'
    max_prompt_tokens: int = 8192
    local_model_path: str = ''


@dataclass(frozen=True)
class DataSettings:
    challenges: str
    solutions: str
    split_seed: int
    exploration: int
    promotion_gate: int
    monitoring: int


@dataclass(frozen=True)
class AgentSettings:
    max_repairs: int
    timeout_seconds: int
    memory_mb: int
    candidates: int
    sandbox: str
    sandbox_image: str = 'arc-sandbox:local'
    cpus: float = 1.0
    pids: int = 32
    max_output_bytes: int = 65536
    max_code_bytes: int = 65536


@dataclass(frozen=True)
class RSISettings:
    exploration_per_generation: int
    min_samples: int
    replay_ratio: float
    trainer: str
    learning_rate: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    promotion_rule: str
    exploration_offset: int = 0


@dataclass(frozen=True)
class TrackingSettings:
    output_dir: str
    profile: str = 'default'


@dataclass(frozen=True)
class BudgetSettings:
    max_seconds: float
    max_gpu_hours: float
    max_llm_calls: int = 100
    max_tokens: int = 100000
    max_candidates: int = 100
    max_container_starts: int = 1000
    token_enforcement: str = "strict"


@dataclass(frozen=True)
class P3BTrainingSettings:
    backend: str = 'none'
    method: str = 'qlora_sft'
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_steps: int = 20
    lora_rank: int = 8
    lora_alpha: int = 16
    continue_previous_adapter: bool = True
    replay_buffer_enabled: bool = True
    max_seq_length: int = 1024
    num_layers: int = 2
    timeout_seconds: int = 3600
    collection_tasks: int = 20
    mlx_lm_version: str = '0.30.7'
    python_executable: str = ''
    model_path: str = ''


@dataclass(frozen=True)
class P3BPromotionSettings:
    enabled: bool = False
    rollback_on_regression: bool = True
    minimum_tasks: int = 5
    diagnostic_task_manifest: str = ''
    exclude_diagnostic_tasks: bool = False


@dataclass(frozen=True)
class P3BInsufficientDataSettings:
    min_verified_samples: int = 5
    min_verified_unique_tasks: int = 5
    action: str = 'skip_training'


@dataclass(frozen=True)
class P3BAblationSettings:
    enabled: bool = False
    max_candidates: int = 3
    learning_rate: float = 0.00001
    step_candidates: list[int] = field(default_factory=lambda: [1, 3, 5])
    canary_max_tokens: int = 128
    max_wall_seconds: int = 1800
    baseline_adapter_path: str = ''


@dataclass(frozen=True)
class P3BSettings:
    training: P3BTrainingSettings = field(default_factory=P3BTrainingSettings)
    promotion: P3BPromotionSettings = field(default_factory=P3BPromotionSettings)
    insufficient_data: P3BInsufficientDataSettings = field(default_factory=P3BInsufficientDataSettings)
    ablation: P3BAblationSettings = field(default_factory=P3BAblationSettings)


@dataclass(frozen=True)
class P3CSettings:
    enabled: bool = False
    task_ids: list[str] = field(default_factory=list)
    diagnostic_task_manifest: str = ''
    max_code_attempts: int = 2


@dataclass(frozen=True)
class SearchSettings:
    """Pre-registered E2/E3 search and collection constraints."""
    enabled: bool = False
    strategies: list[str] = field(default_factory=lambda: ['S0', 'S1', 'S2', 'S3'])
    max_calls_per_task: int = 3
    max_reserved_tokens_per_task: int = 200000
    max_sandbox_starts_per_task: int = 200
    max_seconds_per_task: float = 900.0
    comparison_task_ids: list[str] = field(default_factory=list)
    comparison_task_count: int = 5
    independent_candidates: int = 3
    target_unique_verified_tasks: int = 5
    collection_max_tasks: int = 20
    exclusion_manifest: str = 'configs/p3c.exploration.exclusions.v1.json'
    thinking_policy: str = 'S2_on_others_configured'
    selection_rules: list[str] = field(default_factory=lambda: [
        'train_exact_match_count', 'no_execution_error', 'static_filter_pass', 'candidate_index'])


@dataclass(frozen=True)
class E5Settings:
    """Autonomous RSI protocol. Human-authored settings describe limits, not ARC rules."""
    enabled: bool = False
    profile: str = 'cpu_fake'
    conditions: list[str] = field(default_factory=lambda: ['B', 'C1'])
    generations: int = 2
    real_arc_tasks_per_generation: int = 0
    synthetic_tasks_per_generation: int = 2
    generator_candidates: int = 1
    self_solve_attempts: int = 1
    train_examples: int = 3
    hidden_test_inputs: int = 2
    generator_validation_inputs: int = 2
    minimum_successful_tasks: int = 5
    continue_previous_adapter: bool = True
    replay_ratio: float = 0.5
    replay_max_records: int = 200
    promotion_gate_tasks: int = 1
    promotion_min_delta: float = 0.0
    rollback_on_regression: bool = True
    monitoring_tasks: int = 0
    monitoring_generations: list[int] = field(default_factory=list)
    max_model_calls: int = 100
    max_reserved_tokens: int = 100000
    max_sandbox_runs: int = 1000
    max_training_steps: int = 20
    max_wall_seconds: int = 3600
    max_training_seconds: int = 1800
    checkpoint_dir: str = 'checkpoints/e5'
    log_dir: str = 'logs/e5'


@dataclass(frozen=True)
class E6Settings:
    """Generator-distillation experiment plumbing and finite execution caps."""
    enabled: bool = False
    profile: str = 'cpu_fake'
    e5_artifact: str = 'artifacts/e5-live-smoke-20261101-compact-feedback-final'
    min_unique_verified_tasks: int = 5
    max_collection_attempts: int = 12
    train_examples: int = 3
    query_inputs: int = 1
    heldout_inputs: int = 3
    promotion_gate_tasks: int = 5
    allow_source_mixing: bool = False
    max_model_calls: int = 12
    max_reserved_tokens: int = 100000
    max_sandbox_runs: int = 45
    max_training_steps: int = 0
    max_generations: int = 0
    max_wall_seconds: int = 900
    max_training_seconds: int = 300
    checkpoint_dir: str = 'checkpoints/e6'
    log_dir: str = 'logs/e6'


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSettings
    model: ModelSettings
    data: DataSettings
    agent: AgentSettings
    rsi: RSISettings
    tracking: TrackingSettings
    budget: BudgetSettings
    p3b: P3BSettings = field(default_factory=P3BSettings)
    p3c: P3CSettings = field(default_factory=P3CSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    e5: E5Settings = field(default_factory=E5Settings)
    e6: E6Settings = field(default_factory=E6Settings)


def _settings(cls: type, value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__}: expected mapping")
    known = {f.name for f in fields(cls)}
    required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
    if set(value) - known or required - set(value):
        raise ValueError(f"{cls.__name__}: missing or unknown fields")
    resolved = {}
    for item in fields(cls):
        if item.name in value:
            raw = value[item.name]
        elif item.default is not MISSING:
            raw = item.default
        elif item.default_factory is not MISSING:
            raw = item.default_factory()
        else:
            continue
        if is_dataclass(item.type) and isinstance(raw, dict):
            raw = _settings(item.type, raw)
        resolved[item.name] = raw
    value = resolved
    hints = get_type_hints(cls)
    for name, expected in hints.items():
        item = value[name]
        if expected == list[str]:
            valid = isinstance(item, list) and all(type(x) is str for x in item)
        elif expected == list[int]:
            valid = isinstance(item, list) and all(type(x) is int for x in item)
        elif expected is float:
            valid = type(item) in (int, float)
        else:
            valid = type(item) is expected
        if not valid:
            raise ValueError(f"{cls.__name__}.{name}: invalid type")
    return cls(**value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        value = yaml.safe_load(f)
    groups = {"experiment": ExperimentSettings, "model": ModelSettings, "data": DataSettings,
              "agent": AgentSettings, "rsi": RSISettings, "tracking": TrackingSettings, "budget": BudgetSettings}
    if not isinstance(value, dict) or set(value) - {*groups, 'p3b', 'p3c', 'search', 'e5', 'e6'} or set(groups) - set(value):
        raise ValueError("config must contain the seven required groups and optional p3b/p3c/search/e5/e6 groups")
    parsed = {key: _settings(cls, value[key]) for key, cls in groups.items()}
    if 'p3b' in value:
        parsed['p3b'] = _settings(P3BSettings, value['p3b'])
    if 'p3c' in value:
        parsed['p3c'] = _settings(P3CSettings, value['p3c'])
    if 'search' in value:
        parsed['search'] = _settings(SearchSettings, value['search'])
    if 'e5' in value:
        parsed['e5'] = _settings(E5Settings, value['e5'])
    if 'e6' in value:
        parsed['e6'] = _settings(E6Settings, value['e6'])
    c = ExperimentConfig(**parsed)
    e, m, d, a, r, b = c.experiment, c.model, c.data, c.agent, c.rsi, c.budget
    if any(not math.isfinite(v) for v in (m.temperature, r.replay_ratio, r.learning_rate, b.max_seconds, b.max_gpu_hours)):
        raise ValueError("numeric settings must be finite")
    if not e.name or not m.id or not m.revision or not d.challenges or not d.solutions or not c.tracking.output_dir:
        raise ValueError("required strings must be nonempty")
    if m.adapter_sha256 != 'none' and not re.fullmatch(r'[0-9a-f]{64}', m.adapter_sha256):
        raise ValueError('adapter_sha256 must be none or a SHA-256 digest')
    if not e.conditions or len(set(e.conditions)) != len(e.conditions) or not set(e.conditions) <= {"A", "B", "C"}:
        raise ValueError("conditions must be unique A, B, or C")
    if e.evaluation_split not in {"exploration", "monitoring"} or not e.evaluation_generations or len(set(e.evaluation_generations)) != len(e.evaluation_generations) or e.generations < 0 or any(x < 0 or x > e.generations for x in e.evaluation_generations):
        raise ValueError("invalid generations")
    if any(x < 0 for x in (d.exploration, d.promotion_gate, d.monitoring, a.max_repairs, b.max_gpu_hours)) or sum((d.exploration, d.promotion_gate, d.monitoring)) == 0:
        raise ValueError("invalid split, repair count, or budget")
    if not (1 <= a.candidates <= 2 and a.timeout_seconds > 0 and a.memory_mb > 0 and m.max_tokens > 0 and m.max_prompt_tokens > 0 and 0 <= m.temperature <= 2 and b.max_seconds > 0):
        raise ValueError("invalid agent, generation, or budget limits")
    if not (0 < r.exploration_per_generation and r.exploration_offset >= 0 and
            r.exploration_offset + r.exploration_per_generation <= d.exploration and
            r.min_samples > 0 and 0 <= r.replay_ratio <= 1 and r.learning_rate > 0 and
            r.epochs > 0 and r.lora_rank > 0 and r.lora_alpha > 0):
        raise ValueError("invalid RSI settings")
    if 0 not in e.evaluation_generations and ({"A", "B"} & set(e.conditions)):
        raise ValueError("baseline comparison requires evaluation at generation 0")
    if e.evaluation_split == "monitoring" and d.monitoring == 0:
        raise ValueError("monitoring evaluation requires monitoring tasks")
    if "C" in e.conditions and "B" not in e.conditions:
        raise ValueError("C requires B for the Gen 0 comparison")
    if m.backend not in {'fake', 'openai_compatible', 'ollama_native', 'guarded_mlx'} or a.sandbox not in {'fake', 'docker'}:
        raise ValueError('unsupported backend, sandbox, or trainer')
    if r.trainer not in {'dry_run', 'mlx_lora'}:
        raise ValueError('unsupported trainer')
    p3b = c.p3b
    if p3b.training.backend not in {'none', 'mlx_lora'} or p3b.training.method != 'qlora_sft':
        raise ValueError('unsupported P3-B training backend or method')
    if not (p3b.training.batch_size > 0 and p3b.training.learning_rate > 0 and
            p3b.training.max_steps > 0 and p3b.training.lora_rank > 0 and
            p3b.training.lora_alpha > 0 and p3b.training.max_seq_length > 0 and
            p3b.training.num_layers > 0 and p3b.training.timeout_seconds > 0 and
            p3b.training.collection_tasks > 0):
        raise ValueError('invalid P3-B training settings')
    if type(p3b.training.continue_previous_adapter) is not bool or type(p3b.training.replay_buffer_enabled) is not bool:
        raise ValueError('P3-B training switches must be boolean')
    if p3b.training.backend == 'mlx_lora' and (not p3b.training.python_executable or not p3b.training.model_path):
        raise ValueError('MLX-LoRA requires explicit local Python and model paths')
    if p3b.training.backend == 'mlx_lora' and (
            not re.fullmatch(r'\d+\.\d+\.\d+', p3b.training.mlx_lm_version) or
            Path(p3b.training.model_path).expanduser().name != m.revision):
        raise ValueError('MLX-LoRA requires a pinned runtime and model path matching its revision')
    if (r.trainer == 'mlx_lora') != (p3b.training.backend == 'mlx_lora'):
        raise ValueError('RSI trainer and P3-B training backend must agree')
    if r.trainer == 'mlx_lora' and b.max_gpu_hours < (1 / 3600):
        raise ValueError('MLX-LoRA requires a positive one-second minimum GPU time budget')
    if p3b.training.backend == 'mlx_lora' and p3b.training.collection_tasks > d.exploration:
        raise ValueError('P3-B collection task count exceeds Exploration split size')
    if (type(p3b.promotion.enabled) is not bool or type(p3b.promotion.rollback_on_regression) is not bool or
            type(p3b.promotion.exclude_diagnostic_tasks) is not bool or p3b.promotion.minimum_tasks <= 0 or
            type(p3b.promotion.diagnostic_task_manifest) is not str or
            (p3b.promotion.exclude_diagnostic_tasks and not p3b.promotion.diagnostic_task_manifest)):
        raise ValueError('invalid P3-B promotion settings')
    if (p3b.insufficient_data.min_verified_samples <= 0 or
            p3b.insufficient_data.min_verified_unique_tasks <= 0 or
            p3b.insufficient_data.action != 'skip_training'):
        raise ValueError('P3-B insufficient-data policy must skip training')
    ablation = p3b.ablation
    if (type(ablation.enabled) is not bool or ablation.max_candidates <= 0 or
            type(ablation.learning_rate) not in (int, float) or not math.isfinite(ablation.learning_rate) or
            ablation.learning_rate <= 0 or not ablation.step_candidates or
            any(type(step) is not int or step <= 0 for step in ablation.step_candidates) or
            len(set(ablation.step_candidates)) != len(ablation.step_candidates) or
            ablation.canary_max_tokens <= 0 or ablation.max_wall_seconds <= 0 or
            type(ablation.baseline_adapter_path) is not str):
        raise ValueError('invalid P3-B ablation settings')
    if len(ablation.step_candidates) > ablation.max_candidates:
        raise ValueError('P3-B ablation candidate count exceeds its configured maximum')
    p3c = c.p3c
    if (type(p3c.enabled) is not bool or type(p3c.task_ids) is not list or
            any(type(task_id) is not str or not task_id for task_id in p3c.task_ids) or
            len(set(p3c.task_ids)) != len(p3c.task_ids) or
            type(p3c.diagnostic_task_manifest) is not str or type(p3c.max_code_attempts) is not int or
            p3c.max_code_attempts <= 0):
        raise ValueError('invalid P3-C collection settings')
    if p3c.enabled and (not p3c.task_ids or len(p3c.task_ids) != r.exploration_per_generation or
                        not p3c.diagnostic_task_manifest or p3c.max_code_attempts != a.max_repairs + 1):
        raise ValueError('P3-C requires a fixed task batch, diagnostic manifest, and matching attempt budget')
    search = c.search
    expected_selection_rules = ['train_exact_match_count', 'no_execution_error',
                                 'static_filter_pass', 'candidate_index']
    if (type(search.enabled) is not bool or
            not search.strategies or len(set(search.strategies)) != len(search.strategies) or
            not set(search.strategies) <= {'S0', 'S1', 'S2', 'S3'} or
            (('search' in value) and set(search.strategies) != {'S0', 'S1', 'S2', 'S3'}) or
            min(search.max_reserved_tokens_per_task, search.max_sandbox_starts_per_task,
                search.comparison_task_count, search.independent_candidates,
                search.target_unique_verified_tasks, search.collection_max_tasks) <= 0 or
            not math.isfinite(search.max_seconds_per_task) or search.max_seconds_per_task <= 0 or
            any(type(task_id) is not str or not task_id for task_id in search.comparison_task_ids) or
            len(set(search.comparison_task_ids)) != len(search.comparison_task_ids) or
            search.thinking_policy != 'S2_on_others_configured' or
            search.selection_rules != expected_selection_rules or not search.exclusion_manifest):
        raise ValueError('invalid E2/E3 search settings')
    if search.enabled:
        if (search.max_calls_per_task <= 0 or search.max_calls_per_task > 3 or search.independent_candidates > 3 or
                a.max_repairs + 2 > search.max_calls_per_task):
            raise ValueError('E2/E3 permits at most three total model calls per task and strategy')
        if c.tracking.profile not in {'cpu_fake', 'e2_smoke', 'e2_full', 'e3_smoke', 'e3_collection'}:
            raise ValueError('enabled search requires an explicit CPU Fake, E2, or E3 profile')
        if c.tracking.profile == 'e2_smoke' and search.comparison_task_count != 2:
            raise ValueError('E2 Smoke must use exactly two paired Exploration tasks')
        if c.tracking.profile == 'e2_full' and search.comparison_task_count != 5:
            raise ValueError('E2 Full must use five paired Exploration tasks by default')
        if c.tracking.profile in {'e3_smoke', 'e3_collection'} and search.comparison_task_count != 5:
            raise ValueError('E3 profiles must bind to an E2 Full five-task comparison')
        if (c.tracking.profile == 'cpu_fake') != (m.backend == 'fake'):
            raise ValueError('CPU Fake profile and fake backend must be used together')
        if c.tracking.profile in {'e2_smoke', 'e2_full', 'e3_smoke', 'e3_collection'} and m.backend != 'guarded_mlx':
            raise ValueError('live E2/E3 profiles require the guarded MLX backend')
    elif c.tracking.profile != 'default':
        raise ValueError('an explicit E2/E3 profile requires search.enabled')
    if b.token_enforcement not in {'strict', 'best_effort'}:
        raise ValueError('invalid token enforcement policy')
    if m.backend in {'openai_compatible', 'ollama_native'} and b.token_enforcement != 'best_effort':
        raise ValueError('external server requires explicit best_effort token policy')
    if m.backend == 'guarded_mlx' and b.token_enforcement != 'strict':
        raise ValueError('guarded MLX requires strict token policy')
    if m.backend == 'fake' and a.sandbox != 'fake':
        raise ValueError('fake backend requires fake sandbox')
    if m.backend in {'openai_compatible', 'ollama_native', 'guarded_mlx'}:
        from src.arc.http_backend import OpenAICompatibleBackend
        OpenAICompatibleBackend(model=m.id, revision=m.revision, base_url=m.base_url, temperature=m.temperature,
                                max_tokens=m.max_tokens, seed=e.seed, timeout=m.timeout_seconds,
                                thinking=m.thinking, allow_remote=m.allow_remote, api_style='openai_compatible' if m.backend == 'guarded_mlx' else m.backend, guarded=m.backend == 'guarded_mlx')
        if m.backend == 'guarded_mlx' and m.allow_remote:
            raise ValueError('guarded MLX requires a loopback server')
        e5_local_3gen = bool(isinstance(value.get('e5'), dict) and value['e5'].get('enabled') is True and
                             value['e5'].get('profile') == 'local_3gen')
        if (a.sandbox != 'docker' or (e.generations != 0 and not e5_local_3gen) or
                e.evaluation_split != 'exploration'):
            raise ValueError('real inference requires Docker and an explicitly bounded Gen 0 or E5 3-gen config')
    if not all(math.isfinite(x) and x > 0 for x in (m.timeout_seconds, a.cpus)) or any(x <= 0 for x in (a.pids, a.max_output_bytes, a.max_code_bytes, b.max_llm_calls, b.max_tokens, b.max_candidates, b.max_container_starts)):
        raise ValueError('invalid resource limit')
    if r.trainer == 'dry_run' and r.promotion_rule != "not_applicable":
        raise ValueError("dry-run promotion_rule must be not_applicable")
    e5 = c.e5
    if (type(e5.enabled) is not bool or e5.profile not in {'cpu_fake', 'live_smoke', 'local_3gen'} or
            not e5.conditions or len(set(e5.conditions)) != len(e5.conditions) or
            not set(e5.conditions) <= {'A', 'B', 'C1', 'C2'} or
            ('C1' in e5.conditions and 'B' not in e5.conditions) or
            min(e5.generator_candidates,
                e5.self_solve_attempts, e5.hidden_test_inputs, e5.generator_validation_inputs,
                e5.minimum_successful_tasks, e5.replay_max_records, e5.promotion_gate_tasks,
                e5.max_model_calls, e5.max_reserved_tokens, e5.max_sandbox_runs,
                e5.max_training_steps, e5.max_wall_seconds, e5.max_training_seconds) <= 0 or
            e5.generations < 0 or e5.real_arc_tasks_per_generation < 0 or
            e5.synthetic_tasks_per_generation < 0 or
            (('C1' in e5.conditions) and e5.synthetic_tasks_per_generation == 0) or
            e5.train_examples < 2 or
            not math.isfinite(e5.replay_ratio) or not 0 <= e5.replay_ratio <= 1 or
            not math.isfinite(e5.promotion_min_delta) or not 0 <= e5.promotion_min_delta <= 1 or
            type(e5.monitoring_tasks) is not int or e5.monitoring_tasks < 0 or
            type(e5.monitoring_generations) is not list or
            any(type(generation) is not int or generation < 0 or generation > e5.generations
                for generation in e5.monitoring_generations) or
            len(set(e5.monitoring_generations)) != len(e5.monitoring_generations) or
            type(e5.continue_previous_adapter) is not bool or type(e5.rollback_on_regression) is not bool or
            not e5.checkpoint_dir or not e5.log_dir or
            Path(e5.checkpoint_dir).is_absolute() or Path(e5.log_dir).is_absolute() or
            '..' in Path(e5.checkpoint_dir).parts or '..' in Path(e5.log_dir).parts):
        raise ValueError('invalid E5 settings')
    if e5.enabled:
        if ('C1' not in e5.conditions or
                r.exploration_offset + e5.real_arc_tasks_per_generation * (e5.generations + 1) > d.exploration or
                e5.promotion_gate_tasks * e5.generations > d.promotion_gate or
                e5.monitoring_tasks > d.monitoring or
                e5.max_model_calls > b.max_llm_calls or e5.max_reserved_tokens > b.max_tokens or
                e5.max_sandbox_runs > b.max_container_starts or e5.max_wall_seconds > b.max_seconds or
                e5.max_training_seconds > p3b.training.timeout_seconds):
            raise ValueError('E5 limits must not exceed the enclosing experiment budgets')
        if e5.profile == 'cpu_fake':
            if m.backend != 'fake' or a.sandbox != 'fake' or r.trainer != 'dry_run':
                raise ValueError('CPU Fake E5 requires fake inference, fake sandbox, and dry-run training')
        else:
            if m.backend != 'guarded_mlx' or a.sandbox != 'docker':
                raise ValueError('live E5 profiles require guarded MLX and Docker')
        if e5.profile == 'live_smoke' and (e5.generations != 0 or e5.minimum_successful_tasks <= 1):
            raise ValueError('live E5 smoke must be bounded to Gen 0 with training disabled by its threshold')
        if e5.profile == 'live_smoke' and (r.trainer != 'dry_run' or e5.real_arc_tasks_per_generation <= 0):
            raise ValueError('live E5 smoke requires one Exploration task and disabled training')
        if e5.profile == 'live_smoke' and e5.minimum_successful_tasks <= (
                e5.real_arc_tasks_per_generation + e5.synthetic_tasks_per_generation):
            raise ValueError('live E5 smoke threshold must exceed all tasks that can be collected in Gen 0')
        if e5.profile == 'local_3gen' and e5.generations != 2:
            raise ValueError('local_3gen means Gen 0 through Gen 2')
        if e5.profile == 'local_3gen' and (r.trainer != 'mlx_lora' or b.max_gpu_hours <= 0):
            raise ValueError('local_3gen requires the explicitly configured MLX-LoRA trainer')
        if e5.continue_previous_adapter != p3b.training.continue_previous_adapter:
            raise ValueError('E5 and MLX-LoRA adapter continuation settings must agree')
        if 'C2' in e5.conditions and e5.profile != 'cpu_fake':
            raise ValueError('C2 is limited to the Fake interface tests in this implementation')
    e6 = c.e6
    if (type(e6.enabled) is not bool or e6.profile not in {'cpu_fake', 'live_smoke', 'local_bootstrap'} or
            not e6.e5_artifact or min(e6.min_unique_verified_tasks, e6.max_collection_attempts,
                                      e6.train_examples, e6.query_inputs, e6.heldout_inputs,
                                      e6.promotion_gate_tasks, e6.max_model_calls,
                                      e6.max_reserved_tokens, e6.max_sandbox_runs,
                                      e6.max_wall_seconds, e6.max_training_seconds) <= 0 or
            min(e6.max_training_steps, e6.max_generations) < 0 or
            type(e6.allow_source_mixing) is not bool or e6.allow_source_mixing or
            not e6.checkpoint_dir or not e6.log_dir or
            Path(e6.checkpoint_dir).is_absolute() or Path(e6.log_dir).is_absolute() or
            '..' in Path(e6.checkpoint_dir).parts or '..' in Path(e6.log_dir).parts):
        raise ValueError('invalid E6 settings')
    if e6.max_collection_attempts > e6.max_model_calls:
        raise ValueError('E6 collection attempts exceed its model-call budget')
    if e6.enabled:
        if e6.profile != 'cpu_fake' and e6.min_unique_verified_tasks < 5:
            raise ValueError('Live E6 cannot lower the five-unique-task training gate')
        if (e6.profile == 'cpu_fake' and
                (m.backend != 'fake' or a.sandbox != 'fake' or r.trainer != 'dry_run' or
                 e6.max_training_steps != 0 or e6.max_generations < 1)):
            raise ValueError('CPU Fake E6 requires fake inference, fake sandbox, dry-run trainer, and Gen 0 to 2')
        if e6.profile == 'live_smoke' and (
                m.backend != 'guarded_mlx' or a.sandbox != 'docker' or r.trainer != 'dry_run' or
                e6.max_training_steps != 0 or e6.max_generations != 0):
            raise ValueError('E6 live smoke requires guarded MLX, Docker, zero training, and one generation')
        if e6.profile == 'local_bootstrap' and (
                m.backend != 'guarded_mlx' or a.sandbox != 'docker' or r.trainer != 'mlx_lora' or
                e6.max_training_steps <= 0 or e6.max_generations != 1):
            raise ValueError('E6 bootstrap requires guarded MLX, Docker, bounded QLoRA, and Gen 0 to 1')
        if e6.profile == 'live_smoke' and (
                e6.max_model_calls > 12 or e6.max_reserved_tokens > 100000 or
                e6.max_sandbox_runs > 45 or e6.max_wall_seconds > 900):
            raise ValueError('E6 live smoke exceeds the registered smoke budget')
        if e6.profile == 'local_bootstrap' and (
                e6.max_model_calls > 60 or e6.max_reserved_tokens > 500000 or
                e6.max_sandbox_runs > 180 or e6.max_training_steps > 5 or
                e6.max_wall_seconds > 2400 or e6.max_training_seconds > 1800):
            raise ValueError('E6 bootstrap exceeds its registered live budget')
        if (e6.max_model_calls > b.max_llm_calls or e6.max_reserved_tokens > b.max_tokens or
                e6.max_sandbox_runs > b.max_container_starts or e6.max_wall_seconds > b.max_seconds or
                e6.max_training_seconds > p3b.training.timeout_seconds or
                e6.max_training_steps > p3b.training.max_steps):
            raise ValueError('E6 limits must not exceed enclosing budgets')
        if e6.profile != 'cpu_fake' and m.adapter_sha256 != 'none':
            raise ValueError('E6 starts from the pinned base model; initial adapters are not accepted')
    return c
