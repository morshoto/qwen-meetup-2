# RSI execution boundary: migration procedure

Status: **proposed; no environment has passed**. This document does not
authorize a model call, canary, training or evaluation. The existing v10
registration remains non-execution-ready. Keep all original seals and worktree
changes intact.

## Current host findings

| Requirement | Result | Evidence and scope |
| --- | --- | --- |
| 50/200 filesystem separation | **FAIL** | The combined challenge and solution files under `data/raw` are mode 0644 and their directories are traversable. A different local user alone would not isolate them. No evaluation contents were printed. |
| Alternate paths and IPC | **FAIL for tested design** | The earlier literal path-denial fixture allowed access through a hard link. A deliberately inherited Unix socket remained usable under the network-denying profile. Closing the extra descriptor blocked that fixture only. |
| Complete owned cleanup | **UNVERIFIED** | The monitor records some owned identities, but snapshots can miss descendants, PID start times have one-second resolution, and ports, containers and escaped processes are not covered. On missing group evidence, an exited leader can be reaped while descendants remain. |
| Independent approval | **UNVERIFIED** | A separate AI context reviewed v10 and declined launch; it was not a human or organizational approval. |
| Metal execution in isolation | **UNVERIFIED** | Local Docker reports Linux/aarch64; no existing isolated macOS/Metal execution environment was demonstrated. No model was loaded. |

Existing evidence: [scoreboard](resume-scoreboard-20261106.md),
[review](resume-independent-context-review-20261106.md),
[decision log](end-to-end-decision-log-20261106.md), and owner-only local
`artifacts/rsi-resume-v10-review-index.json`. The review index points to
separately held raw inputs; it is not an approval or a complete launch closure.

## Concrete migration path

1. An authorized operator provisions a dedicated Apple-silicon Mac or an
   existing macOS guest with a **separate evaluator and generation trust
   boundary**. A physically separate offline Mac is the simpler candidate.
   Do not put the combined challenge file, solutions, evaluation tasks,
   private probes, backups, shared folders, cloud sync or credentials on the
   generation machine. Keep evaluation assets on the evaluator side.
2. An evaluator, in a separate trusted preparation environment, verifies the
   reviewed split and creates a generation-only export. Transfer only that
   export, the reviewed generation code and configuration, pinned local model
   files, and required offline dependencies. Bind each transfer to a reviewed
   manifest and digest. Do not copy the repository wholesale: it contains the
   combined data. The existing export/loader is provisional and still needs
   independent review and a launch binding.
3. Use an operator-provisioned standard execution account and an
   OS-enforced environment that cannot access evaluator storage or services.
   Check owner, mode, ACL, group membership and inherited descriptors for
   each mounted resource. Disable shared directories, networking and access to
   evaluator-side services; do not pass inherited sockets or file descriptors
   except reviewed standard streams. Inspect other IPC and service channels.
   A separate account is insufficient if source data or copies remain world
   readable, and a path denylist is insufficient for aliases and hard links.
4. Independently test the candidate boundary with **synthetic sentinel files**
   and a cooperating outside service, not the actual 50/200 questions. Cover
   direct paths, symlinks, hard links, alternate mounts, inherited regular
   file/socket descriptors, network and local IPC. Record the exact attempted
   paths, account, policy, expected and observed denials without exposing
   private contents. Check actual source mounts and ACLs separately. This
   testing is required for both generation and training launchers.
5. Implement an owned execution supervisor with a separately verifiable
   containment boundary. Register process and container identities at launch;
   test normal exit, timeout, crash, parent exit, orphaning, PID reuse,
   permission failure, partial observation, listener shutdown and container
   cleanup. Verify absence from an independent authority after teardown.
   Where a disposable VM is used, validate the host supervisor's lifecycle,
   guest storage, ports and helper processes; guest termination alone is not
   sufficient evidence. Failed or incomplete observations remain unverified.
6. Before using a guest for real work, separately demonstrate Metal **compute**
   and the installed MLX runtime with synthetic data. Apple documents
   Metal-accelerated guest graphics and macOS virtualization, but those facts
   do not prove this particular guest supports MLX compute or 14B/32B memory
   requirements. If that check fails, use a suitable physical Mac instead.
   [Apple virtualization documentation](https://developer.apple.com/documentation/virtualization/virtualize-macos-on-a-mac),
   [Apple paravirtualized graphics documentation](https://developer.apple.com/documentation/paravirtualizedgraphics).
7. Freeze a new model-specific launch closure, budget and boundary evidence;
   obtain the required independent review of those exact hashes and decisions.
   Only after the review and all applicable gates pass can the already
   authorized local synthetic real-weight canary be considered. Canary
   adapters remain nonpromotable. Oracle collection and RSI training retain
   their separate scientific gates.

Creating accounts, changing permissions, installing a VM or transferring
artifacts was not performed here. The earlier plan did not authorize sudo,
downloads, cloud transfer, public release or destructive changes. The operator
must provide the already provisioned environment and applicable authorization.

Apple describes both App Sandbox access controls and separate POSIX/ACL
controls; configuring one does not demonstrate the complete boundary above.
[Apple sandbox file access](https://developer.apple.com/documentation/security/accessing-files-from-the-macos-app-sandbox).
