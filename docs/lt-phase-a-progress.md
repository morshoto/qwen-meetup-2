# LT Phase A: exploratory ARC comparison

Status: **incomplete; no model calls or ARC evaluation performed**. This is a
separate protocol from the strict RSI experiment in issue #2. No frozen task
manifest, model-specific run manifests, real predictions, or comparison score
exists yet. The comparison, attestation and scoring utilities are offline scaffolding,
not an execution gate or a complete live runner.

## Local inventory (2026-11-06)

- The issue #2 worktree was clean at `03653fadf47739fa9ffa9c02a044539a875c58b6`;
  its changes have been merged as PR #7. Existing seals and artifacts were not
  modified. Its source data files remain local and are not copied into this branch.
- The local training challenge SHA-256 is
  `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`;
  the separate solution file SHA-256 is
  `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9`.
  These hashes identify local files; they do not independently authenticate origin.
- The earlier E9 record identifies 14B revision
  `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` and 32B revision
  `bcaaf7f538adf166c1080a2befdb4f6019f66639`. Reverify the installed model
  files, runtime and template before any new run.
- Docker reports image `arc-sandbox:local` as
  `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`.
  Image availability alone does not verify containment or cleanup.
- `research/rsi/bounded-exposure-proposal.json` and related reviews inventory
  prior project use, but explicitly leave historical and pretraining exposure
  uncertain. No 200 IDs have been selected for this protocol.

## Remaining work before a comparison

1. Complete the development-only real-model path and verify model, template,
   budget, sandbox and process cleanup on the actual host. The existing
   [execution-boundary review](../research/rsi/execution-boundary-migration-20261106.md)
   records unresolved cleanup and evaluator-isolation findings. Fixture tests do
   not resolve them. Stop and record an incomplete run if controls cannot be
   established.
2. Audit exposure and select development, Phase B training and 200 comparison
   IDs without overlap. Freeze task contents, source files, prompt, runtime,
   model revisions, seeds, candidate count and all resource budgets. The
   current helpers bind proposed selection, configuration and supplied input,
   model and runtime inventories. They do not authenticate provenance, prove
   those inventories are complete, or establish execution readiness.
3. Build a bounded live runner, model-specific manifests, and durable prediction
   and resource records. The `run_task` and `run_bounded_task` helpers accept
   injected components and are fixture seams, not safe live launchers. Tests
   check held-out answers do not affect agent prompts or repair feedback; this
   does not prove operating-system isolation.
4. Run models sequentially with equivalent settings only after those controls
   are established. Report actual attempts, failures, denominators and costs;
   do not represent an incomplete run as a completed comparison. Do not use
   the official 120-task evaluation set here.
5. Freeze Phase B's training design before inspecting the 200-task results if
   those tasks will be reused for its comparison; otherwise disclose reuse as
   exploratory. No training or Phase C evaluation is included here.

## Offline scoring

`python -m src.arc.lt_score_cli --frozen FROZEN.json --receipt INPUTS.json
--challenges CHALLENGES.json --solutions SOLUTIONS.json --sources SOURCES.json
--prompt PROMPT --results RESULTS.jsonl` writes a JSON report to stdout only
after rechecking the frozen selection and input receipt. `SOURCES.json` maps
stable inventory names to local file paths. Each result row must include the
frozen `comparison_sha256`, model key (`14b` or `32b`), task ID, status,
candidates, tokens, runtime and peak memory/swap values; use JSON null for
unavailable measurements. The report hashes the input results and explicitly
marks run identity as independently unverified. Preserve the original result
JSONL separately because the scoring report does not embed predictions.

Task pass@2 requires one complete candidate to match every test output. For
output-level pass@2, each output counts if either candidate matches that output;
those matches can come from different candidates. Denominators reflect only
available candidate slots, with missing tasks and candidates reported explicitly.
A reported incomplete result must not be interpreted as a 200-task score.
