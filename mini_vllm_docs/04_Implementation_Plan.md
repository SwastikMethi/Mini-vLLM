# Mini vLLM Implementation Plan

Version 1.1  |  17 October 2026

Build a verified model runner first, then paged storage and scheduling, then the request-driven UI. The two-day target is a constrained core demonstration on a ready device. A complete release with robust replay, API handling, UI polish, and controlled benchmarks is estimated at five to seven focused developer days. Prefix reuse and preemption are separately gated extensions.

## Starting assumptions

One developer; one local inference device; downloaded Qwen3-0.6B weights; Python 3.11; Node for the React UI; no login or deployment work. CUDA is the performance target. CPU and MPS are compatibility paths that require their own parity checks. Device availability, model download, and environment setup can extend the schedule.

| Milestone | Estimate | Exit evidence |
| --- | --- | --- |
| M0 Environment and reference | 0.5 day | Locked versions, model revision, device manifest, and reference logits |
| M1 Paged runner | 1 to 1.5 days | Real cache tensors and correct single/batched outputs across block boundaries |
| M2 Scheduler and lifecycle | 0.5 to 1 day | Continuous decode batches, admission credits, cancellation, and no leaked blocks |
| M3 Events API and replay | 1 day | Typed events, atomic snapshot, resumable SSE, and replay parity |
| M4 Request UI | 1 to 1.5 days | Three linked views: lifecycle, logical/physical KV map, and batch timeline; replay and error states |
| M5 Validation and measurements | 1 day | End-to-end checks and reproducible benchmark report |

## Two day core target

Day one: establish the reference, implement the pool and attention adapter, and pass short single-request parity checks. Day two: add batched decode, conservative FIFO admission, typed in-memory events, and a minimal live request view. Deliver one real multi-request trace with allocation and cleanup. Treat durable journal replay, detailed memory inspection, and benchmark conclusions as subsequent work if the core integration consumes the time box.

## Scope protection

Keep R0 prefix sharing and preemption disabled; prove memory safety with capacity credits. Keep prefills bounded and complete, and preserve batched decode as a real model call. Defer kernel work until the Python reference passes correctness checks. If there is no compatible device, label the device limitation explicitly and validate tiny CPU workloads rather than substituting a fake inference animation.

## Implementation tasks and dependencies

| Task | Depends on | Implementation and acceptance |
| --- | --- | --- |
| T01 Reference harness | M0 | Load pinned model; disable thinking; record tokenizer and config; compare logits and deterministic test generations. |
| T02 KV allocator | T01 | Atomic allocation, free list, ownership, block tables, valid lengths, capacity credits, and boundary tests. |
| T03 Attention adapter | T02 | Preserve Q/K norm, RoPE, GQA, masks, and weights; gather valid histories; compare every sampled decode position. |
| T04 Batch runner | T03 | Pack one token per active row; preserve positions and masks when membership changes; verify against independent references. |
| T05 Scheduler | T04 | FIFO admission with growth credits; one short prefill between decode iterations; reasons for waiting and impossible requests. |
| T06 Lifecycle | T05 | EOS, limits, deadlines, cancellation, failure, and cleanup exactly once; stable output indices. |
| T07 Event contract | T02 to T06 | Implement schema validation and event emission alongside mutations; verify no invented phases. |
| T08 Journal and API | T07 | SQLite transaction boundaries, snapshots, idempotent submission, SSE cursor recovery, and export. |
| T09 UI reducer | T07 | Build shared selections and all three projections from single- and multi-request fixtures; isolate fixture mode from live runs. |
| T10 Live UI integration | T08 and T09 | Bind stages, queue reasons, slot commits, logical/physical links, time-scaled batches, reconnect, cancel, and cleanup. |
| T11 Load and release checks | T10 | Run benchmark matrix, trace-fidelity suite, accessibility checks, and measured UI delay. |

### Suggested repository layout

backend/mini_vllm contains config, request, kv_cache, attention, runner, scheduler, engine, events, journal, and server modules. backend/tests contains reference, allocator, model, scheduler, and stream checks. frontend/src contains the API client, event reducer, request workspace, KV inspector, batch view, replay controls, and shared theme tokens. benchmarks contains workload fixtures and a runner. contracts contains the shared schema and examples.

Generate TypeScript event types from the validated backend schema, or validate both against the same checked-in JSON Schema. Changes to event fields require a schema version and fixture update before either side merges.

## Validation release and extension backlog

### Release verification matrix

| Test | Expected evidence |
| --- | --- |
| Reference parity | Prompts at 15, 16, 17, 31, 32, and 33 tokens; several decode steps; dtype tolerances and divergence investigation |
| Request isolation | Different prompts and lengths in one batch match isolated runs; no cross-request cache reads |
| Capacity and cleanup | Waiting queue, impossible request rejection, cancellation in every state, and full return of ownership |
| Event fidelity | Monotonic seq; queue updates preserve entry time; correct slot commits; cross-selection and replay equal the final snapshot |
| Recovery | Duplicate delivery, cursor gap, reconnect after terminal, stale run, slow subscriber, and journal failure |
| Performance | Identical seeded arrival trace, warmup, repeated runs, timeout accounting, and raw results |

### Benchmark plan

Begin with concurrency 1, 2, 4, and 8; prompt lengths around 32, 256, and 1024 tokens; and output caps of 32 and 128. Include burst arrivals and paced arrivals. Record actual prompt/output lengths, early EOS, accepted/rejected counts, and all failures. Collect at least 1000 completions per condition before emphasizing p99; preliminary samples are explicitly exploratory. Compare reference, static batching, and continuous paged execution with the same trace mode.

### Definition of done

- Fresh setup instructions reproduce a live request on the documented device. Lockfiles, model revision, defaults, and limitations are included.

- A three-request demonstration shows a newcomer prefill between older decode steps, a shared decode batch, a capacity wait, safe cancellation, and replay parity across all three views.

- The UI contains no timer-driven lifecycle transitions or fixture values in live mode. Metrics identify their clock and scope.

- Output correctness, cache ownership, event ordering, reconnect, and cancellation gates pass. Measurements include overhead and unsuccessful requests.

### Next milestones

R1a prefix reuse: key complete blocks by exact prefix and model identity; add references, immutable sharing, copy-on-write, and cached-prefix visualization. R1b preemption: switch to a documented overcommit policy, preserve output and RNG state, rebuild K/V on resume, and show recomputation as real work. R2: profile the gather bottleneck, implement direct paged attention, then evaluate chunked prefill. Re-run the same correctness and trace gates after each change.
