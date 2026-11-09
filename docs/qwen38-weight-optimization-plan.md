# Qwen3.8 exploratory weight optimization implementation plan

Issue #16 is a new experiment; the earlier local comparison and Phase B registrations are immutable. First establish a bounded, independently invocable synthetic canary against the pinned 27B snapshot and toolchain. Record command, full log, model hashes before and after, loss, adapter changes and reload/logit evidence, resource observations and cleanup; a failed or incomplete check blocks collection. Synthetic success is not ARC improvement.

Next register a seeded fresh public Training pool and separate validation set, excluding the existing 50, official Evaluation, previous reservations, known exposures and matching content/lineage. Freeze identities, inputs, prompts, budgets and incomplete allocation before generating. Test overlap, immutable registration and leakage boundaries first.

Collect ordinary full-task prompts breadth first, preserving every raw response and attempt; independently verify against permitted solutions offline and deduplicate parent/content lineage. Audit historical records with source provenance and the same exclusions. Test that only actual verified model outputs enter training and that validation answers never become targets.

After the canary, freeze the small-data gate and training configuration. Train only with at least five verified unique eligible parents after token filtering; otherwise report insufficient data without a score. Verify adapter hashes, tensor changes and reload/logit deltas. If trained, evaluate paired adapter-off/on on the existing exploratory 50 and report incomplete denominators, gains/losses and source composition. Do not claim recursion or generalization from a parameter update.

## Collection implementation status

The new path includes full-task request/capture, breadth-first scheduling, a private append-only attempt journal, a supervised one-request worker, offline verification receipts, and pinned-canary bundle replay. Synthetic tests exercise their data flow, timeouts, resource stops and orphan cleanup without ARC examples or model inference. The collection command requires the separately frozen [execution descriptor](../research/rsi/qwen38-weight-execution-v1.json) and [bounded lineage review](../research/rsi/qwen38-weight-bounded-review.json); both were prepared before collection. The empty supplied lineage register and unknown semantic and pretraining exposure remain explicit limitations. Fresh model collection has been dispatched in the registered private run directory. Source-bound historical replay, SFT preparation, real-data training and paired evaluation have been implemented; the latter stages still require a terminal real collection and their own evidence. Reconciliation of an interrupted live worker remains a manual process. The previously scored comparison and historical proxy collector remain unchanged.

## Frozen collection policy

The reviewed proposal predeclares 40 fresh parents and 25 separate validation parents, seed 20261111, and the existing direct-grid prompt at temperature 0.6 with a 4096-token completion limit and 18432-token context. The scheduler attempts candidate 0 across all fresh parents, then candidate 1 across all parents. It then takes up to six additional breadth-first attempts only for parents without a verified success; it stops on completion, the registered 24-hour wall limit, the 6,553,600 reserved **completion-token** ceiling, or a resource/process failure. Prompt tokens are separately recorded and are not part of that completion-token reservation. Each attempt reserves 4096 completion tokens even on error. The per-attempt worker has a 600-second ceiling, a 10% minimum sampled system-free-memory threshold, a 16 GiB swap ceiling, and a 2 MiB diagnostic-output threshold. The wall limit is anchored to one immutable local run directory and includes time between resumed steps after that run starts.

The separate validation parents are never LoRA targets, and their Gen 0 success is not required. After collection and token-length filtering, at least five independently verified unique training parents are needed for the exploratory adapter. If fresh data are too sparse, the predeclared fallback audits eligible original FP8 outputs, retaining their distinct `historical_fp8` identity; a mixed adapter is cross-run verified-sample fine-tuning. Fewer than five qualifying parents yields `insufficient_verified_data`, without an invented score. The previously evaluated 50 remain exploratory paired comparison tasks, not independent validation or official evaluation.

The real-data trainer is predeclared with a 3600-second training limit and a separate 900-second adapter-reload probe, using the same sampled 10% free-memory and 16 GiB swap limits as collection. Training requires replay of the terminal fresh journal and verifier receipt, any needed historical source, the private SFT bundle, its source inventory, and the exact trainer tokenization and completion mask. It saves initialized and trained LoRA tensors, requires finite reports for all 20 steps, and only records a parameter update after a separate finite-logit reload probe and a final source replay. An interrupted or failed training attempt keeps its evidence and is not automatically repeated. These controls have synthetic tests. The actual trainer processor has also produced a completion-masked synthetic example; real-data length filtering and the adapter probe still need verification after collection. No real-data adapter has been trained.


## Running the post-collection stages

Run these commands from the issue worktree with the pinned Python environment. The private run directory is fixed by the execution descriptor; moving or changing it requires a new registration. The worktree archive `arc-prize-2025.zip` contains the registered `arc-agi_evaluation_challenges.json`; its challenge-file hash was checked against the frozen proposal. The CLI reads that challenge file for exclusion checks; it does not read official evaluation solutions. The following local paths show the registered environment and may be adjusted only where identities still match the frozen inputs.

```bash
TASK_REPO="$(git rev-parse --show-toplevel)"
TASK_MAIN="/Users/shotomorisaki/Engineering/qwen-meetup-2"
TASK_PYTHON="$TASK_MAIN/artifacts/qwen38-local-env/venv/bin/python"
TASK_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
TASK_RUN="$TASK_REPO/artifacts/qwen38-weight-exploratory-20261109-v1"
TASK_HISTORY="$TASK_MAIN/artifacts/qwen38-quick-20261108/qwen38-quick-phasea-v1-fp8-hopper"
TASK_COMPARISON="$TASK_MAIN/artifacts/qwen38-local-20261109"
TASK_ARCHIVE="$TASK_REPO/arc-prize-2025.zip"
TASK_ARGS=(
  --manifest "$TASK_REPO/research/rsi/qwen38-weight-execution-v1.json"
  --proposal "$TASK_REPO/research/rsi/qwen38-weight-allocation-proposal.json"
  --review "$TASK_REPO/research/rsi/qwen38-weight-bounded-review.json"
  --repo "$TASK_REPO" --model-path "$TASK_MODEL"
  --challenges "$TASK_MAIN/data/raw/arc-agi_training_challenges.json"
  --solutions "$TASK_MAIN/data/raw/arc-agi_training_solutions.json"
  --archive "$TASK_ARCHIVE"
  --canary-dir "$TASK_MAIN/artifacts/qwen38-issue16-canary-capture-20261109"
  --run-dir "$TASK_RUN" --history-dir "$TASK_HISTORY"
)
```

Wait for collection to finish or exhaust its predeclared budget. The fresh replay rejects a live, interrupted, or resource-stopped collection; inspect the original worker and journal rather than retrying an ambiguous started attempt. Then prepare the private bundle and independently replay it:

```bash
"$TASK_PYTHON" -m src.cli.qwen38_weight_prepare_sft "${TASK_ARGS[@]}"
"$TASK_PYTHON" -m src.cli.qwen38_weight_prepare_sft "${TASK_ARGS[@]}" --verify-only
"$TASK_PYTHON" -m src.cli.qwen38_weight_train "${TASK_ARGS[@]}"
```

Preparation selects only independently verified, length-eligible outputs. It consults historical FP8 records only when fresh examples do not meet the five-parent gate. The training command records `insufficient_verified_data` instead of training when that gate fails. If training fails or is interrupted, preserve its private artifacts and investigate rather than automatically rerunning.

Only after the training result reports `trained` and its adapter evidence replays, freeze and run the paired evaluation, then create the offline report:

```bash
"$TASK_PYTHON" -m src.cli.qwen38_weight_evaluate prepare "${TASK_ARGS[@]}" --comparison-dir "$TASK_COMPARISON"
"$TASK_PYTHON" -m src.cli.qwen38_weight_evaluate run "${TASK_ARGS[@]}" --comparison-dir "$TASK_COMPARISON"
"$TASK_PYTHON" -m src.cli.qwen38_weight_evaluate report "${TASK_ARGS[@]}" --comparison-dir "$TASK_COMPARISON"
```

The evaluation journal and report retain incomplete runs and errors. Open [the results notebook](../nb/results/17_qwen38_weight_optimization.ipynb) from this worktree to display the available receipts without exposing candidate responses. Report creation, evaluation and training are not implied by a successful synthetic canary.

### Recorded recovery for this run

The first registered evaluation stopped on its first worker before model loading: the saved adapter directory was mode `755`, while the worker required a private directory. The failed start, worker logs, execution receipt and `resource_stop` termination remain under `evaluation/adapter-v1`. No response was produced. The directory was made private without changing the adapter bytes. A separately registered `evaluation-v2` recovery binds the exact original failure evidence and retains the same comparison, adapter, prompt, seeds and limits. It does not replace or resume the stopped journal.

For this recorded recovery, use `src.cli.qwen38_weight_evaluate_recovery` in place of `src.cli.qwen38_weight_evaluate` in the three evaluation commands above. Its separate descriptor is `evaluation/evaluation-v2.json`; its private journal and reports are under `evaluation/adapter-v2`. The recovery is specific to the recorded pre-inference failure and rejects altered failure evidence or any original response.
