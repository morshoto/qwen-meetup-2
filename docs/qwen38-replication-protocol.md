# Frozen Qwen3.8 LoRA replication protocol

This study compares the #16 frozen Qwen3.8-27B MLX 4-bit base with the same base plus its mixed-source LoRA adapter on the 25 previously reserved validation parents. The bounded exposure audit checks recorded reservations, generation and training parents, exact visible inputs, known lineage and answer fingerprints. The lineage register is empty, and unrecorded semantic and pretraining exposure remain unknown. The study is **exploratory**; the earlier 50-task comparison was already examined and is not the baseline for these new tasks. Official evaluation challenges are exclusion inputs only, and official evaluation solutions are not accessed.

Freeze the descriptor before new predictions or decoding any reserved answers. The two arms use the same direct-grid prompt, format recovery, temperature 0.6, two recorded candidates per task, 4096-token generation limit, 18432-token context, and seeds `20261111 + candidate_index`. Run the base arm over all registered tasks first, followed by the adapter arm, using the same task order. The complete run reserves 409,600 completion tokens, 24 hours wall time, 600 seconds per attempt, a minimum 10% free-memory threshold and a maximum 16 GiB swap threshold. Resource stops, errors, and unresolved starts must remain visible; an unresolved start requires reconciliation, not silent retry.

The primary comparison is the paired difference in whether either recorded candidate is correct. The null is equal probabilities of paired gains and losses; use an exact two-sided sign test with alpha 0.05. Report first-candidate and either-candidate counts, validity, duplicate raw pairs, gains, losses, shared outcomes, complete paired denominator, resource evidence and runtime. Missing or failed work is incomplete, never silently incorrect. The two attempts are not assumed independent. Do not retune the prompt, model, adapter, checkpoints or protocol using these results. A single mixed-source fine-tuning round is not demonstrated recursive improvement; the bounded audit cannot establish complete semantic or pretraining non-exposure.

The registration binds the original #17 revision, adapter and base identity, the bounded audit receipt, source code hashes, data hashes, per-task visible fingerprints, execution settings, and the private run directory. A supervised worker receives only the visible challenge request and model identity. Read Training solutions only for offline scoring after a terminal run and successful receipt replay. Keep original #16 artifacts unchanged.

## Replay the offline report

The completed run and source inventory are private, ignored local artifacts. Replay requires the original issue 16 worktree and its unchanged training/evaluation artifacts, the pinned model snapshot, training challenge and solution files, the challenge-only official exclusion export, and the pinned Python environment. Paths below reflect this run's registered machine; identities are verified and missing or drifted inputs fail closed. This reads Training answers for offline scoring after completion. It does not start inference or access official evaluation solutions.

```sh
TASK_MAIN=/Users/shotomorisaki/Engineering/qwen-meetup-2
TASK_REPO="$PWD"
TASK_RUN="$TASK_REPO/artifacts/qwen38-replication-v1"
TASK_OLD="$TASK_MAIN/.codex/worktree/16/artifacts/qwen38-weight-exploratory-20261109-v1"
TASK_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
"$TASK_MAIN/artifacts/qwen38-local-env/venv/bin/python" -m src.cli.qwen38_replication report \
  --repo "$TASK_REPO" --model-path "$TASK_MODEL" --old-root "$TASK_OLD" \
  --execution "$TASK_REPO/research/rsi/qwen38-weight-execution-v1.json" \
  --proposal "$TASK_REPO/research/rsi/qwen38-weight-allocation-proposal.json" \
  --history "$TASK_MAIN/artifacts/qwen38-quick-20261108/qwen38-quick-phasea-v1-fp8-hopper/manifest.json" \
  --snapshot "$TASK_RUN/source-inventory.json" --audit "$TASK_RUN/bounded-audit.json" \
  --challenges "$TASK_MAIN/data/raw/arc-agi_training_challenges.json" \
  --official "$TASK_RUN/official-challenges.json" \
  --solutions "$TASK_MAIN/data/raw/arc-agi_training_solutions.json" --run-dir "$TASK_RUN" \
  --report "$TASK_OLD/evaluation/adapter-v2/report-023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70.json" \
  --adapter-path "$TASK_OLD/training/adapter-run-v1/adapter"
```

Run from the issue 18 worktree. The public [results](qwen38-replication-results.md) preserve the summary and receipt identities if the private artifacts are unavailable.
