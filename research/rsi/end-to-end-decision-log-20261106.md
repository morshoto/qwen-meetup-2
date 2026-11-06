# End-to-end RSI execution: decision log

## 2026-11-06: Phase 0 inventory

Scope: local 14B and 32B execution under the user's end-to-end objective.
Worktree `issue/2-rsi-execution` was clean at commit
`49c922e5f74b04c48838130cafc9fc877482c4fe` when inspected. Existing sealed
artifacts were not modified. This log is an inventory, not a protocol lock,
independent review, or authorization to bypass a gate.

- Objective attachment SHA-256:
  `bae933d09996fb31a7768771e65b70cc585375fe7bb14b1ee3256c18a19b322d`.
- Main worktree `PLAN.md` SHA-256:
  `8a30ae2308ed84b77241b555803a7724b09e6a5c20ce2d2f10c0b030da56b332`.
  It describes phases P0–P6 and the local execution ceilings. The separately
  named `RSI_14B_32B_Autonomous_Goal_Prompt.md` was not found in the supplied
  attachment directory or either worktree. Its absence matters because the
  objective delegates budget, safety-stop and retry rules to both files.
- No `AGENTS.md` was found in either worktree or the checked parent locations.
- Model identities recorded in the existing canary drafts: 14B revision
  `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`; 32B revision
  `bcaaf7f538adf166c1080a2befdb4f6019f66639`. These are recorded identities,
  not a new verification of weight files.
- Existing environment `/tmp/qwen-p3a-mlx-venv`: MLX 0.32.4, MLX-LM 0.30.7,
  Transformers 5.22.1, huggingface-hub 2.4.0. The shell's default `python3`
  does not contain MLX; use the recorded environment for any future work only
  after rechecking its identity.
- Local Docker image `arc-sandbox:local` resolved to
  `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`.
  This observation alone does not prove sandbox isolation.
- At 2026-11-06 04:31 UTC, the host reported 64 GiB physical RAM, 63 GiB
  available disk and zero configured swap in `vm.swapusage`; this is a
  point-in-time observation, not training feasibility.
- Allocation review SHA-256
  `435db21430293e904f4c0074468a6bcfc953cfb4ad49beeb6bbaebc87468b024`
  records a provisional 750/50/200 split and 317 known exposures in the
  generation pool. It explicitly says independent review is incomplete and
  execution is not ready. Unknown project and pretraining exposure remain.
- The existing 14B and 32B Oracle generation drafts have SHA-256 respectively
  `f6996bd06766254bdbecf8b005b0eb83d91d5036c43c13534f231797faaaf384` and
  `c996706f91fa2a604668944706af206e37f7d3887bdc3e862ac351b79421ac3a`.
  They are drafts and their stricter registered budgets cannot silently be
  replaced by the new plan's larger ceilings.
- Canary v22 draft SHA-256 values: 14B
  `12f9215104e4a1f91e52fb7fbba31c5c7d26361ac5d943c09b78b230414ac717`;
  32B `56ad72708b8212f1decf0c2385b131492082781f11e7a703672728a5074fe327`.
  Both remain unverified and not execution-ready. The new objective provides
  local execution authorization, but does not waive independent review or
  verified cleanup.

Decision: preserve the provisional split and legacy candidate records without
promoting them. Obtain the missing referenced prompt, then reconcile its rules
with the plan. Continue technical cleanup work only within the stated hard
stops; do not start live generation or model canaries until their applicable
independent review, containment and registration requirements are met. The
existing candidate inventories and reviewer packets are indexed in
`bounded-review-guide.md`, `oracle-probe-review-guide.md` and
`technical-canary-review-guide.md`.

## 2026-11-06: Canary IPC containment fixture

An owned synthetic child under the previous `(allow default) (deny
process-fork)` profile successfully connected to an owned Unix socket
listener; a regression fixture also showed successful loopback TCP connection.
This provides a concrete counterexample to treating fork restriction as
complete prevention of external delegation. No model or training ran.

The proposed profile now also contains `(deny network*)`. A local Darwin
fixture observed `PermissionError` for both Unix-domain and loopback TCP
connections to owned listeners. The new regression first failed for both
protocols on the old profile; after the change, 36 focused containment,
monitor, group-observation and canary tests passed. Existing thread and fork
fixture checks also passed. These are narrow checks, not proof that all IPC,
Mach services, external process creation, or model compatibility are covered.
The EPERM and cleanup questions remain open; independent review is still
required. The source change makes the v22 source inventory and its derived
canary registration stale: regenerate the review packet and recheck all hashes
before considering any launch. Do not reuse its previous identity as current.

## 2026-11-06: Synthetic MLX and generation-input checks

With the network-denying profile, the opt-in fabricated tiny Qwen-shaped MLX
fixture completed one synthetic training step and reloaded 28 matching saved
adapter factors. No Qwen weights or ARC tasks were used. This supports only
compatibility with that tiny fixture; it does not establish model feasibility,
cleanup or containment. A direct comparison of each v22 source inventory
against its 264 recorded files found `src/arc/rsi_canary_containment.py` as a
changed file before the additional change below.

The generation stage previously accepted a task mapping containing IDs outside
its generation manifest, although it selected only registered slots for
dispatch. A synthetic regression showed that an extra held-out task could be
passed into the stage. The stage now rejects extra IDs before writing its start
record or calling the transport. The regression failed before the change; the
generation-stage and audit suite then passed (11 tests; the separately included
opt-in tiny test was skipped in that invocation). This is input validation,
not filesystem isolation: the generation process may still be able to read a
combined source dataset or other local files. Enforced separation, manifest
provenance, independent review and the missing prompt remain open before any
live collection. This source edit also invalidates affected generation and
canary source inventories; rebuild and review them before use.

## 2026-11-06: Filesystem isolation investigation

A synthetic Darwin sandbox fixture created an owned file, a hard link to the
same inode, and a separate allowed file. A profile with `(deny file-read*
(literal ...))` denied the named file, but the sandboxed child could read its
hard link and the separate file. The fixture used temporary synthetic content,
no ARC data, and no model. This demonstrates that a deny rule for the known
combined dataset path cannot by itself establish evaluation-data isolation.

Decision: do not add a path denylist and claim isolation. A prospective
generation launcher needs a separately prepared generation-only input and an
independently reviewed boundary that also covers other copies, aliases and
process access; the full source dataset must not be handed to that launcher.
The manifest check added above helps reject accidental extra task mappings but
does not establish that boundary. The current fork/network profile is a
technical-canary fixture, not a generation isolation policy.

## 2026-11-06: Generation-only export preparation

Implemented a deterministic in-memory export that first rechecks the existing
allocation, validates generation grids and exact example fields, and returns
only the 750 generation tasks with visible training examples and test inputs.
Its receipt binds the source, selection, generation manifest and export
digests and explicitly marks isolation and execution unverified. Synthetic
tests check exclusion of both held-out partitions, absent test answers, digest
binding, deterministic output, and rejection of changed inputs. The focused
export, allocation and generation-stage suites passed 18 tests. The initial
export test failed before implementation because the module did not exist.

No real export was created or handed to a model. The preparation process still
reads the combined challenge source to validate the allocation, so it must be
separate from the generation process; the function alone does not enforce
that separation, prevent access to other local copies, review the split, or
authorize a run. A reviewed launcher, storage permissions and isolation
boundary remain necessary. This new code must be included in fresh source
inventories and independent review.

The companion generation-side loader now accepts only export bytes and a
generation ID manifest, checks both digests and the exact ID set, validates
the task schema and grids, and constructs `Task` objects without opening the
combined dataset. Synthetic tests exercise tampered bytes, a changed manifest,
and a test answer inserted with a recomputed export digest. The focused export,
stage and allocation suites passed 19 tests. The loader is not connected to a
reviewed live launcher and does not restrict the caller's other file access;
it does not establish isolation or authorization.

The synthetic export and loader were also exercised through the existing
generation stage with a fake transport and fake sandbox. The stage completed
its synthetic slots and retained `training_allowed: false`; no real model or
ARC data were used. The broad `unittest discover -s tests/arc` run completed
with 654 tests passing and one skipped. It was started before the additional
integration test was written; that test passed in a separate invocation.
These checks do not provide independent review, live provenance or process
isolation evidence.

## 2026-11-06: Allowlist feasibility probe

An additional synthetic macOS profile denied file reads by default and
allowed selected runtime directories and a synthetic input directory. Its
first attempt could not execute the virtual-environment interpreter; after
adding the virtual environment to the allowed paths, the interpreter exited
with signal 6 before reporting file-read outcomes. Both attempts used only
temporary synthetic files and no model. The specific missing runtime access
or other cause was not established. This profile is not adopted: neither
successful MLX execution nor denial of alternate paths was demonstrated. A
reviewable allowlist must identify its dependencies and be tested against
aliases and IPC without weakening the evaluation-data boundary.

Follow-up controls: with the same broad allow rules, both `deny file-read*`
and `deny file-read-data` caused the virtual-environment Python to exit with
signal 6; removing the file denial allowed it to start. Changing the working
directory to the allowed virtual environment did not help. The Homebrew
interpreter, `/usr/bin/true` and `/bin/echo` also exited with signal 6 under
the data-denial profile. This narrows the failure to the profile/runtime
interaction rather than the application or its current working directory,
but does not identify the kernel-level cause or prove deny/allow precedence.
The profile remains unusable and no security conclusion follows from it.

## 2026-11-06: Current source review snapshot

Created owner-only local `artifacts/rsi-local-source-review-20261106.json`
(SHA-256 `05f76b66b18924a665ba478474d164dbca5b515f9626a1723d2a4fc6dc5e3e4e`)
with the current 265-file source inventory and hashes of the three changed or
new focused test files. It includes the untracked export module. It is a
review input, not a replacement launch registration, a freeze, or independent
review. Recompute with `build_source_inventory` and hash the three test files
before review; any later source change invalidates the snapshot. It contains
hashes and paths, not probe grids or evaluation content. The missing prompt,
model and dependency evidence, process cleanup and isolation require separate
review; the snapshot does not resolve them.

## 2026-11-06: Self-contained recovery plan and current registration draft

The recovery plan and both previously requested appendices are available in
the main worktree's `PLAN.md`; no further attachment is required. Execution
remains in `issue/2-rsi-execution`, and the existing uncommitted files are
preserved. A separate read-only reviewer context was asked to inspect the
split, provenance, containment, cleanup and current source identities. It is
not a human or external organizational reviewer.

Created new owner-only `artifacts/rsi-resume-registration-20261106-v1/` with
`protocol.lock` SHA-256
`6a547adc1644cec3f5fbe493b7e48c0c2e2541b7cdc8f2dd2ceda2aad5b56b1b`,
`budget.lock` SHA-256
`cd3bb1b39ccfec5404bc71d5f102dfdb0600ca291a041bcd5c02070c4802e317`, and
source inventory SHA-256
`b4ba1f1ad500c4a63f1283f1936ef98ac13a6100a1fe8e0ffcf5c945984209c2`.
It binds current source, provisional allocation/draft identities, recorded
model revisions, package versions and local image identity. It is deliberately
non-authorizing, not a complete launch closure, and preserves the stricter
existing pilot limits. The older v22 source and launch hashes are stale; no
old seal was modified. Resource observations and all launch identities must
be rechecked at execution time.

## 2026-11-06: Local model-file recheck

Rehashed all 12 files in the retained 14B model manifest (8,323,870,520
bytes) and all 14 files in the retained 32B manifest (18,445,874,457 bytes).
Both matched their respective prior local manifests, with zero unreadable or
mismatching files. This took approximately 10.9 seconds and did not load
weights or run inference. The owner-only report is
`artifacts/rsi-resume-registration-20261106-v1/model-file-comparison.json`,
SHA-256 `0fa2428389639a4a48ac50b63569393b1d4d92663c740d554e5c946f8ed20562`.
This is a point-in-time comparison to previously local manifests, not
independent publisher authentication, launch-time identity, memory feasibility,
or independent approval. It does not update the already written protocol lock.

## 2026-11-06: Observed descendant identity tracking

Added a fixture-only tracker that keeps previously observed descendant
identities after reparenting and identifies reuse of an observed PID with a
different caller-supplied start token. It does not signal any process, and all
snapshots explicitly leave cleanup and escaped-process exclusion unverified.
The test was first run before the module existed; then 20 focused tracker,
process monitor, group and sampling tests passed. This is not a live launcher:
the observer is not integrated with OS sampling, ports, container identities,
or independently authenticated start times. Children that start and escape
between samples remain invisible. The source change invalidates the v1 source
inventory and protocol packet; create and review a new packet before any use.

An owner-only successor draft was created at
`artifacts/rsi-resume-registration-20261106-v2/`, with the 266-file inventory
SHA-256 `fc76d12a10df4d715279e60bc9e24ead1be377a216f39a15dedd6eebda56da6c`
and protocol SHA-256
`abc471f9a969f6b15dbbf9006a383edbd18e9b868a89751e860aa77f7f501d40`.
It supersedes the stale v1 source snapshot but is itself non-authorizing and
has not received independent review.

## 2026-11-06: Owned process start observation

The descendant tracker now has a bounded read-only macOS `ps` observation
adapter. It confirms the root is an owned, live, unreaped child session leader
using `waitid`, PGID, session ID and parent checks, then captures PID, parent,
reported start time and state. An actual harmless child fixture was observed;
an unavailable `ps` fixture failed closed without signalling it. Twenty
focused tests passed. The `ps` start time has only second resolution; it is not
a cryptographic identity or proof against all PID reuse. This observer does
not yet integrate with the canary launcher, record ports/containers, account
for all descendants between samples, or certify cleanup. No model was loaded.

The changed code supersedes the v2 source inventory. New owner-only v3 draft
`artifacts/rsi-resume-registration-20261106-v3/protocol.lock` has SHA-256
`1113e37a7272097cd14c31fb843c09f9bf63de6124b11ac32cd560191fc58ea5` and
source inventory SHA-256
`afbfee09cd85cf63c253a9166f458442dc1b184c4167d5b0abfc883f7f57e5f7`.
It remains unreviewed and not execution-ready.

## 2026-11-06: v10 review and full test run

The separate reviewer rechecked all 266 v10 inventoried files and the bound
budget and input hashes. They confirmed the owned exited-child fallback does
not signal an unverified group, but noted that it cannot account for surviving
descendants. Isolation and cleanup remain unverified; neither model is approved
for canary or Oracle collection. See `resume-independent-context-review-20261106.md`.
An owner-only supplementary inventory of 11 relevant test files is at
`artifacts/rsi-resume-v10-test-manifest.json`, SHA-256
`7f00d1bd35259a34c3af1e5f3198dca588b24cb15cb5335380176241b01b403f`;
it is not bound into the existing protocol lock.

The current full ARC suite completed 674 tests with one skipped in 85.84
seconds. Its local output is `/tmp/rsi-resume-v10-full-suite.log`, SHA-256
`30b451fc9f5baefbb596ff75dd0cfee5b700e2483e549e00eba5e5f2d16d4dae`.
These tests do not establish live model feasibility, independent safety review
or verified cleanup. No real model or evaluation task was run.

## 2026-11-06: Local isolation environment inventory

The current host reports macOS 26.5.1 and hypervisor support. The `tart`,
`multipass`, `limactl` and `podman` commands were not present on the current
PATH. The available Docker daemon reports Linux/aarch64; no claim is made that
it can run the macOS Metal-backed MLX workload. These observations do not
prove that no other VM, account or isolation mechanism exists, and no VM was
created. Without an available verified isolated environment or privilege
changes, the current generation-only export and sandbox fixtures do not
establish the required live filesystem boundary. The plan prohibits sudo,
downloads and cloud migration for this run.

## 2026-11-06: Local review index

Created owner-only `artifacts/rsi-resume-v10-review-index.json` with SHA-256
`19172df566408479370fe0321bcd67fc5fd06db1bb914891b03e2193b7a20791`.
It indexes hashes for the v10 protocol, source and budget, test inventory,
local model-file comparison, allocation and historical summaries/seals,
separate-context review and review guides. It also records hashes and access
metadata for the four locally retained probe files without embedding their
contents. All four were regular non-symlink files owned by the current user
with mode 0600 at observation time. The index is local review material, not
a complete audit or launch registration. Historical raw logs and private
review inputs remain separately stored, and access controls and hashes can
change; a reviewer must recheck them. No external transfer or model run
occurred.

## 2026-11-06: Exited leader and unavailable group observation

The separate reviewer found that an exited owned child could remain unreaped
if group observation failed. The fixture now reaps the child already confirmed
exited by `waitid` in that case, without signalling an unverified group, and
returns `leader_reaped_group_unverified`. Potential descendants remain
unaccounted for and cleanup is not verified. The new failure-path test failed
before the change; 27 focused fixture/monitor tests passed afterward. A broad
suite begun before this last source change passed 673 tests with one skipped;
it does not test the final changed source. No Qwen model or ARC task ran.
The successor owner-only v10 protocol draft is
`artifacts/rsi-resume-registration-20261106-v10/protocol.lock`, SHA-256
`db924fb27e05c0edfb80b6e325eef1412601d6d71a37bef3606189874ab0a4a8`.
It remains unreviewed and not execution-ready.

## 2026-11-06: Identity sampler diagnostic

Eight bounded harmless-child fixture runs collected 56 identity observations;
all eight stopped on normal child exit and no sampler or tracker exceptions
were captured. The earlier intermittent `identity_monitor_unavailable` did
not recur in this small sample, so its cause remains unknown. This does not
establish absence of a race or prove cleanup. No model was loaded. A separate
resume scoreboard records per-model status without treating an unstarted
Oracle search as a measured zero yield.

## 2026-11-06: Watchdog identity-sampling integration

The owned-process watchdog and training-stream wrapper now accept an opt-in
identity sampler. A missing, malformed or timed-out sample stops the owned
process; a reported reused PID also stops it. A harmless real child fixture
was observed using the macOS tracker and a synthetic failure was observed to
stop the training-stream fixture. Thirty-three focused monitor, tracker,
training/reload stream and group-observation tests passed. This is optional
fixture integration, not a reviewed registered launcher: no code claims that
the supplied sampler is authenticated or required by all callers. Sampling
cannot detect every escaped process and does not cover ports or containers;
cleanup remains unverified. No model was loaded.

Owner-only v4 draft `artifacts/rsi-resume-registration-20261106-v4/protocol.lock`
has SHA-256 `8b11450a25d1c31a0b44d04eedaa4a336e1957c14ff8e59f10ad8995d00a4c00`
and its 266-file source inventory SHA-256 is
`5f03915635d25d3012546c6c769bb22daadcd42c5ca75cb45b00b44d86a887dd`.
The draft remains unreviewed, non-authorizing and not execution-ready.

## 2026-11-06: Tiny training and reload identity samples

The reload stream now passes an optional identity sampler to the watchdog;
the synthetic failure test stops the reload child when the sampler is
unavailable. The opt-in fabricated tiny MLX pipeline uses the owned-process
observer for both its one synthetic training step and adapter reload, and
records identity sample counts and initial identity observations. The tiny
fixture initially failed its new identity-observation assertion before
integration. Afterward, the tiny fixture and related monitor, training,
reload and tracker suites passed 32 tests. No real Qwen weights or ARC data
were used. This is not a model canary, an independent resource/identity audit,
or proof of cleanup. Ports, containers, escaped processes between snapshots,
and second-resolution PID identity remain unresolved.

Owner-only v5 draft `artifacts/rsi-resume-registration-20261106-v5/protocol.lock`
has SHA-256 `74bb875eafeb673b2f82cbb416ec1524781f58d5b2c37a65b3dee63e0dc9a888`
and its source inventory SHA-256 is
`961d5a6ca9ea6320d2eadae075c24925c0e8e6e9d144c08b5b7d681e3b773f35`.
It remains unreviewed, non-authorizing and not execution-ready.

## 2026-11-06: Inherited-descriptor isolation fixture

A synthetic owned Unix socketpair showed that a descriptor deliberately
passed into the fork/network-restricted child remained usable: the child sent
`fixture` despite `deny network*`. With that extra descriptor closed, the
same attempted write failed with bad-file-descriptor and no message arrived.
The fabricated tiny training and reload `Popen` calls now specify
`close_fds=True` explicitly (the Python default already closed extra
descriptors). Eighteen focused containment, tiny MLX and monitor tests passed.
This does not prove standard streams, other launchers, Mach IPC or external
services are safe; it exposes a condition the eventual reviewed launcher must
control. No real model or ARC task ran.

The owner-only v6 draft
`artifacts/rsi-resume-registration-20261106-v6/protocol.lock` SHA-256 is
`56f8d47fcadbac808f61e28239f6ffe5b3e868fa5032121a44d09081dd20afa7`;
its source inventory SHA-256 is
`dd91d90a87425c3cee8a01a24106259de24c6fe52124a5330bc7d1015892468b`.
It is not independently reviewed or execution-ready.

## 2026-11-06: Exceptional fixture cleanup

The tiny fixture's exceptional cleanup no longer calls `poll()` before
considering the process group. It checks ownership through unreaped-child
observation, checks live session/PGID or the exited leader's group listing,
then uses the existing bounded group-stop path; it never reports cleanup as
verified. Focused tests cover a live child, exited unreaped child, repeated
cleanup, and rejection of a non-session child. A live identity-sampling test
once stopped with `identity_monitor_unavailable` rather than normal exit; a
subsequent run observed normal exit. The test now accepts either fail-closed
outcome and continues to require observed samples and unverified cleanup. The
underlying intermittent sampling failure has not been diagnosed. Thirty-six
focused cleanup, tiny MLX, process, tracker, training and reload tests passed.
No real Qwen weights or ARC data ran. Sampler threads that outlive timeouts,
escaped descendants, ports and containers remain unverified. Source changes
make v7 stale and require a fresh reviewed registration before any launch.
The v8 owner-only provisional draft is
`artifacts/rsi-resume-registration-20261106-v8/protocol.lock`, SHA-256
`f15485759f2af219bcab1d2bf1dba98853046ba6ea3b1570fae960de7519340a`;
it remains unreviewed and not execution-ready.

## 2026-11-06: Timed identity sample and tracker mutation

For the tiny fixture path, the timed worker now collects raw `ps` rows without
mutating the tracker; the watchdog applies returned rows to the tracker on its
own thread only after the worker returns within its timeout. A test delays the
worker beyond its deadline, waits for it to finish, and confirms the tracker
processor was never invoked; this test failed before the change. The tiny MLX
fixture and relevant cleanup, tracker, monitor and stream suites passed 37
tests. Generic caller-supplied samplers can still have arbitrary side effects;
the worker itself cannot be cancelled. The tracker still cannot establish
cleanup or detect unsampled escaped processes, and reload is not run after
unverified training cleanup. No real Qwen model or ARC task ran.

The owner-only v9 draft
`artifacts/rsi-resume-registration-20261106-v9/protocol.lock` has SHA-256
`ea4d8a2190f7db2df70e478d7a07b82a4b9751186835f1d0fc0b24bbb8d3e52d`;
its source inventory SHA-256 is
`c13ed9b6ea6409b7ef57c8d352ec4491721553abd9c6c4702df03398d6367d63`.
It remains unreviewed and not execution-ready.

## 2026-11-06: Real reparenting fixture

An owned, harmless fixture parent started a child with a fixed two-second
lifetime. The tracker observed the descendant before the parent exited, then
still reported that previously observed PID as live after the parent was
reaped and the descendant reparented. No descendant was signalled. The
tracker and monitor focused suites passed 21 tests. This checks one sampled
reparenting sequence; it does not prove the descendant's termination, exclude
unsampled children or establish verified cleanup. No model or ARC data ran.
Only a test file changed; the v6 source inventory excludes tests, so the
changed test's identity must be included in any complete future review packet.

## 2026-11-06: Reviewer finding: reload before cleanup

A separate read-only reviewer found that the tiny fixture started its reload
subprocess after training despite unverified cleanup. The fixture now stops
before reload unless group cleanup and escaped-process exclusion are verified.
Since current monitoring cannot establish those facts, the fixture records
`not_run_cleanup_unverified`; the new expectation failed before the fix and
25 focused tests passed after it. An older tiny reload success remains
historical evidence only and did not meet this boundary. The same review found
that `poll()` in exceptional cleanup may reap the group leader too early,
and a timed-out daemon sampler may still mutate its tracker. Both require
additional work. v6's source inventory is stale after the fix; do not use it
for execution. No model or ARC task ran.

The successor owner-only v7 draft is
`artifacts/rsi-resume-registration-20261106-v7/protocol.lock`, SHA-256
`4afbbc1aa38dde37e636bfacbf7a7e8ea5ee707bd06f3580a9214cfcdab1696f`.
Its source inventory SHA-256 is
`c547abad412d5398a22d3f80243bd07e1a9acf756340028a0ee2929526369938`.
It is not independently reviewed or execution-ready.
