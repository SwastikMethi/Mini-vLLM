# Benchmark notes (R0, exploratory)

Run `20261017-200725` — `benchmarks/run.py --quick`: Apple M1 Max, **MPS bf16**, torch 2.14.1, transformers 5.20.1,
Qwen3-0.6B@c1899de, greedy, EOS-or-cap stop, burst arrivals, 8 requests per condition, one warm-up pass of the
identical workload per condition. Raw rows, manifest and full table: `benchmarks/results/20261017-200725/`.

**Exploratory only**: 8 completions per condition (the release bar for a reported p99 is ≥1000). Treat numbers as
direction, not results.

| Condition | HF one-at-a-time | HF static batch | Continuous paged (R0) |
| --- | --- | --- | --- |
| c=1, prompt 32, cap 32 — out tok/s | 43.0 | — | 25.1 |
| c=4, prompt 32, cap 32 — out tok/s | 42.4 | 146.2 | 92.4 |
| c=4, prompt 256, cap 32 — out tok/s | 33.9 | 117.1 | 66.4 |
| c=4, prompt 32 — E2E p95 ms | 5445 | 1580 | 2179 |

## Findings

1. **Continuous batching beats one-at-a-time** at concurrency 4 (2.0–2.2× output tokens/s, ~2.5× lower E2E p95):
   requests share decode forwards (mean decode batch 3.6–3.8).
2. **R0 is slower than HF static batching on this workload**, and slower than HF one-at-a-time at concurrency 1.
   A burst of equal-cap requests is static batching's best case: no late arrivals for continuous batching to exploit.
3. **The gap is the gather attention path, not engine overhead.** Isolated decode step at 64 tokens of history:

   | Batch | Paged R0 decode | HF cached decode | Slot-index build |
   | --- | --- | --- | --- |
   | 1 | 29.4 ms | 20.2 ms | 0.4 ms |
   | 4 | 39.4 ms | 21.5 ms | 1.3 ms |

   Continuous ITL p50 at c=1 is 29.5 ms — equal to the bare paged forward, so scheduling, events and
   detokenization add ~nothing. On MPS a decode step is launch-bound (HF is flat from B=1 to B=4); R0 adds per-layer
   gather, zeroing and masking kernels and pads every row to the longest history.
4. **MPS latency artefacts**: kernels compile on the first forward of a new shape (first B=3 decode: 458 ms), and the
   first forward after the GPU idles for minutes ran 1.4–2.0 s for a 20-token prefill that otherwise takes 50 ms.
   The benchmark warms up on the identical workload and has no idle gaps; live-UI TTFTs after a pause include this.

## Not yet measured

Paced arrivals, 1024-token prompts, cap 128, concurrency 2/8, ≥1000-completion runs, CPU and CUDA. Tracing on/off
overhead was not separated (the engine always emits events; per finding 3 it is within noise here).

## Implication

This matches the TRD's R0 expectation ("can run slower than optimized inference libraries"). The R2 direction is a
direct paged-attention read (no gather/pad) — re-run this matrix before and after with the same manifest.
