# P3-C E2/E3 Live start audit

Audit date: 2026-11-01 (Asia/Tokyo). This records the worktree before the
live-protocol changes. Existing staged and unstaged work is preserved.

## Baseline evidence

- Branch: `main`; the worktree already contained staged, unstaged, and
  untracked P0–P3-C work. No reset or cleanup was performed.
- The live profile points at
  `mlx-community/Qwen3-8B-4bit`, revision
  `545dc4251c05440727734bcd94334791f6ab0192`, 4-bit, no adapter, temperature
  0.7, 2,048 requested tokens, seed 42, and guarded MLX on `127.0.0.1:8080`.
- The ARC Training split is 400 Exploration / 400 Promotion Gate / 200
  Monitoring. Its canonical split SHA-256 is
  `104e1ee7abab600ed8d4c49a4c01ec8f679351ad4931e6e089b5c0ca980afe09`, equal
  to the retained P3-C offset-285 run. The split was not regenerated or
  edited.
- The checked-in exclusion registry contains 287 previously used/diagnostic
  IDs. It includes all five H1 format-diagnostic IDs, the P3-C offset-285
  batch, and prior P3-B runs. CPU Fake artifacts use synthetic IDs and are
  separate from the ARC Training corpus. No completed E2-Live or E3-Live run
  was present at audit time.
- E1's retained 20-task artifact reports 0/20 independently verified,
  1/36 fully matching train examples, 39/39 valid Python candidates, and 59
  parsed responses. Historical request/token totals are run-level because the
  original schema did not bind each request to a task.

## Gaps requiring correction before live dispatch

- E2 selected a search strategy using held-out `pass_at_1`, contrary to the
  train-only selection rule. The saved fake comparison also ranked S1 and S3
  using the same train-match objective.
- There was one five-task live profile, but no separate two-task Smoke and
  five-task Full gates, no pre-dispatch machine-readable policy artifact, and
  no E2 Resume protocol.
- E2 emitted a frozen config for incomplete and CPU Fake runs. E3 did not
  require a completed E2-Full live selection before live collection.
- The existing E2/E3 artifacts did not include all requested snapshots,
  task/attempt manifests, budget ledger names, comparison JSON, or artifact
  seals. The per-task request ceiling was four, allowing an extra analysis
  call beyond the requested three-call cap for S1/S3.

## Host readiness

- The pinned Qwen snapshot exists locally. The native arm64 runtime at
  `/tmp/qwen-p3a-mlx-venv/bin/python` has MLX-LM 0.30.7; its installed
  `generate_step` SHA-256 matches the reviewed value
  `1aee463145a78556a66604ace7b6d85dc10035226aa4f909cbefccfa81b49de6`.
- The ordinary project `uv` interpreter is x86_64 under Rosetta and has no
  MLX package; it is not the inference runtime.
- Docker Desktop was initially stopped. It has since started, and the local
  `arc-sandbox:local` image is available with ID
  `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`.
- No guarded server was listening on port 8080 during the audit. No live model
  request had been issued.

The pre-change CPU Fake runs are orchestration evidence only. They are not
Qwen performance evidence and are not eligible to unlock E3-Live.
