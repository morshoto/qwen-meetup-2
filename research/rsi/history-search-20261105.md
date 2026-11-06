# Search for additional Qwen experiment history

This is a bounded local investigation, not a completeness certificate or an
audit of model pretraining. No model call or training was performed.

## Locations checked

- Compared the path and SHA-256 of every regular file in `artifacts/` across the
  three `qwen-meetup-2`, `qwen-meetup-2-issue-1`, and `qwen-meetup-2-issue-2`
  worktrees. Each had 7,775 artifact files; neither other worktree had an extra
  or different file relative to issue 2 at the time of this check.
- Searched shallow names in local Engineering research directories, Downloads,
  Documents, Desktop and temporary directories. The E8 and E9 documents in
  Downloads are experiment instructions, not a separate run archive. This was
  not an exhaustive search of every disk or cloud service.
- Searched local Codex session files for the original repository path. Two
  older session files identify that repository as their session working
  directory. Their SHA-256 digests at review time were:
  - `796e4dd7a4b3ea4960fc956d3d97d822e2eed7a305ff8b4c5baaa686490b2f20`
    (`2026-10-31T15-24-41`, session `01a190a1-bc28-7b53-8065-f7ddd6bc7df8`)
  - `2bdf25a668f80530f715e7a121e2a9f3198edaf912ea46e3be94e8e6e30e9860`
    (`2026-11-01T13-13-20`, continuation `01a1954f-d8a1-7d22-9d06-4bc0676aa1d3`)

## Findings

The older session logs contain command output from existing artifact
locations, including `p3b-local-b20-792fca0afd82` and
`p3b-local-qwen8b-b20-offset260-07182ba11e74`. Their retained `data_split.json`
files each enumerate all 1,000 tasks, so they include all 200 provisional
development IDs. Their retained `results.json` files each contain 20 rows and
no development task IDs. Listing a task in a split manifest is not evidence that
it was sent to a model; the results check is limited to these two runs.

The session continuation references a planned `e8/live-smoke-32b-retry-20261101`
campaign directory that is absent from the current artifacts. Its captured
preparation output reports `download_incomplete` and `inference_started: false`
after a download timeout. This observation covers that preparation attempt,
not every possible later action.

No separate historical prompt archive was identified in these locations. The
original logs can contain truncated outputs, reference paths that were never
created or have since disappeared, and do not prove that every prior run was
retained. The provisional split remains unverified and the protocol's stop
condition remains in force. See the [experiment report](../../docs/rsi-experiment-results.md)
for the other gates.
