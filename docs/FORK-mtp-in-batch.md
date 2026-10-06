# Fork design note: MTP drafts in batch windows (issue #857)

Goal: let `--batch N` slots decode with MTP drafts, so `"parallel"` stops losing throughput
(measured on 2× RTX 5090 / IQ3_S / 1M pool: per-request 40 tok/s and aggregate 0.82× of solo; see
issue #857 comment by gucasbrg, 2026-10-06).

## Hard limits that bound the design (verified at 6f32ec0)

- `kVerifyMaxT = 8` (`include/strata/kernels/verify_kernels.hpp:21`) — rows per window.
- MoE window entries: `tokens*top_k <= 128` (`src/core/expert_source.cpp:1978-1993`), top_k = 10.
- Batch contract: "Greedy only, no drafts (MTP) in a batch window" (`verify.hpp:141`).

Feasible slot×draft combos inside both caps (block per slot = 1 fed + K drafts):

| slots × K | rows | entries | fits |
|---|---|---|---|
| 4 × 1 | 8 | 80 | yes |
| 2 × 3 | 8 | 80 | yes |
| 2 × 4 | 10 | 100 | no (rows) |
| 4 × 4 | 20 | 200 | no (both) |

Start with **4 × 1** (smallest change).

## Junctions that hard-code one-token-per-row (all at 6f32ec0)

- `BSlot::x` scalar + `sl.x = y; sl.p += 1;` — `src/program/generate.cpp:5882, 6034-6035, 6118-6119`
- `stage_batch` one token+pos per row — `src/core/verify.cpp:1910-1918`
- commit record written as keep-1 — `src/core/verify.cpp:1928-1936` (`c[0]=1; c[1]=0;`)
- commit graph launched with no host decision (pipeline) — `src/core/verify.cpp:2065-2072`
- `MtpDrafter` is a process-wide singleton bound to the main session, one draft K/V — `generate.cpp:3004,3016`; `src/core/mtp.cpp:150-311`; full layer refuses T≠1 (`mtp.cpp:469`)
- accept scan is host-side prefix compare, a local `int a` — `generate.cpp:7136-7160, 8050`

Reusable machinery: multi-row MoE gather (`ExpertJobMulti`, MAXT=8), verifier slots
(`init_slots/run_slot_rows/commit_slots`), per-slot sessions/sampling/penalties, conversation
snapshots (incl. the draft-K/V-as-last-image convention), `copy_to_slot`, the server-side
solo↔slot swap, coupled-draft counter arithmetic, `serve/test_parallel.py` as the acceptance harness.

## Milestones

- **M1** `MtpDrafter` → multi-state: N draft K/V states + per-state staging/graph keys; multi-row
  full layer (lift the T≠1 guard).
- **M2** verifier: per-slot `(1+K)`-row blocks in a batch window (row→slot mapping + graph key),
  host-side accept for the **non-pipelined** `batch_step` path first, per-slot commit `n_keep`
  (the commit record already has the solo path's `n_keep`+position-array shape; batch fills only
  index 0 today).
- **M3** `generate.cpp` `batch_step`: assemble blocks, host accept, advance a per-slot draft queue,
  emit `BT` per accepted token.
- **M4** functional test on ONE GPU, small context: parity — a slot's greedy tokens must equal its
  solo greedy tokens (contract in `verify.hpp:141-152`); then measure.
- **M5** pipeline path (`pump`/`batch_launch`/`batch_poll`): device-side accept (write `n_keep`
  into `h_commitb_` from a tiny kernel) to keep the no-host-decision property; dual-card real test.
- **M6** widen K where caps allow (2 slots × 3 drafts).

## Dev loop (node-110)

- Source: `/ssd/software/strata-fork` (this fork, branch `f-batch-mtp`).
- Build: `bash /ssd/software/fork-build.sh` (docker `strata-dev`, SM120, ~8 min with -j; log
  `/ssd/software/fork-build.log`). Network: GitHub needs the **192.168.18.210:7890 proxy**.
- Test: swap `build/strata` over `/opt/strata/engine/strata` in a run of the `strata` image
  (same compose env, prod container stopped for the window).
