# R0 release check — 2026-10-17

Apple M1 Max, 32 GB, macOS; MPS bf16 (live, UI, benchmarks), CPU fp32 (parity and engine tests).
Qwen/Qwen3-0.6B@c1899de, torch 2.14.1, transformers 5.20.1, React 19 / Vite 8.

## Automated gates

| Gate | Evidence | Result |
| --- | --- | --- |
| Model output parity | `test_paged_parity.py`: 7 prompt lengths (15/16/17/31/32/33/65) × 8 greedy steps, block size 16 and 4; mixed batch with mid-batch removal and late join; MPS bf16 tolerance test | pass; CPU max dev 9e-5, MPS 0.42 = HF's own cached-decode deviation ([PARITY.md](PARITY.md)) |
| Memory lifecycle | `test_allocator.py`, `test_engine.py`: credits, all-or-nothing growth, reuse, release once; every terminal path (complete, cancel queued/active, failure, deadline) ends with zero owners and zero credits | pass |
| Trace fidelity | `test_fixtures.py`: all three contract traces replay to their checkpoint/terminal snapshots; `test_engine.py`: every emitted event validates against `event_schema.json`, seq is dense, replay == live snapshot | pass |
| Concurrency | `test_engine.py`: B prefills between A's decode steps, A+B share a decode batch, C queues for `capacity` and is admitted only after a release, engine intervals never overlap | pass |
| Streaming/recovery | `test_server.py`: Last-Event-ID resume without duplicates, Last-Event-ID beats after_seq, snapshot + after_seq, stale cursor → `reset`, slow subscriber dropped, journal failure → `engine.failed(trace_complete=false)` + 503, restart → new run, old run replays | pass |
| UI reducer | `frontend` vitest (22): fixture snapshots, duplicate/gap handling, slots never fill on `token.sampled`, replay rate and idle-gap skip, replay ends equal to live | pass |
| UI responsiveness | Headless Chromium, production build, 8 concurrent requests, 2763 events: event-receipt → painted frame **p50 9.9 ms, p95 14 ms, max 18 ms** (target p95 < 250 ms). Excludes server serialization and network (same host). | pass (after fix below) |

Totals: backend `pytest` 41 passed; frontend `vitest` 22 passed; `tsc -b` clean.

## PRD release demonstration (live, MPS)

1. **Long prompt prefill, first token before any decode** — live: prefill batch emits `token.sampled` with the
   prefill batch ID before the first decode batch; the KV grid shows the prompt's blocks committed at prefill
   completion. ✔
2. **Requests joining an active decode; shared batch; capacity wait** — live: three prompts sent back-to-back share
   decode batch `b09` (3 members). Capacity wait is **not** shown live with the default 256-block pool (it needs
   a small pool); it is proven in `test_engine.py` (pool 6) and in the `?fixture=batching` replay. ◐
3. **Cancel one request; blocks released only after the in-flight step** — live: RQ-006 cancelled mid-decode from the
   detail drawer → `request.cancel_requested` → `cache.released` (3 blocks, 18 credits) → `request.terminal
   (cancelled, user)`. ✔
4. **Disconnect and reconnect without duplicate tokens** — live: stream dropped, RQ-007 submitted and completed while
   offline, Reconnect → browser watermark 1903 = server 1903, journal contiguous, 0 duplicates, RQ-007 equal to
   the server snapshot on every field. Server restart: page shows "Not connected", then follows the new run. ✔
5. **Replay at reduced speed equals live final state** — 0.25×/0.5× replay rate and final-state equality verified
   deterministically in vitest; live replay ran in the app, but the browser pane was hidden so its wall-clock pacing
   was not observable there. ✔ (state) / ◐ (visual pacing)
6. **Export trace and benchmark manifest** — `GET /api/runs/{id}/export` (manifest with device/versions/
   `trace_mode: live`, initial state, ordered events, final snapshot) tested in `test_server.py`;
   `benchmarks/results/20261017-200725/manifest.json` identifies device, versions, workload and measurement mode. ✔

## Issues found and fixed during the check

- KV grid pushed the timeline off-screen with the real 256-block pool → panes now scroll locally.
- Timeline rendered the entire run every frame (54,645 SVG nodes, 1.06 M px) → windowed to the viewport; p95
  render delay 376 ms → 14 ms.
- Browser stayed on a dead/stale run after a server restart (proxy 502 ends EventSource retries; retained old run
  looks healthy) → reconnects go through `/api/health`, plus a 5 s health watchdog.
- Status strip said "Live inference" while disconnected → shows the connection state.
- Replay crawled through idle gaps at original spacing → idle gaps with no live request are skipped (presentation only).
- After a journal failure the `engine.failed` notice was rejected by the projection as a seq gap → seq continues
  from the last committed event.

## Known limits (not R0 blockers)

Performance: R0 continuous paged is slower than HF static batching on burst workloads; the gap is the gather
attention path ([BENCHMARKS.md](BENCHMARKS.md)). Benchmarks are exploratory (8 per condition). GPU durations are
unavailable (host timings only). MPS shows cold-start / idle-wake latency spikes on first forwards. CUDA untested.
