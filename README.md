# Mini-vLLM

A local learning inference server: real **Qwen3-0.6B** generation over a **paged KV pool** with **continuous
batching**, and a UI that renders only what the engine actually did — request lifecycle, logical→physical KV
blocks, and a time-scaled batching timeline, all from one ordered event journal.

Specs live in [mini_vllm_docs/](mini_vllm_docs/README.md). Parity evidence: [docs/PARITY.md](docs/PARITY.md).
Release check: [docs/RELEASE_CHECK.md](docs/RELEASE_CHECK.md).

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), Node 20+. Tested on an Apple M1 Max (MPS, bf16); CPU (fp32)
is the parity path. CUDA is untested here.

```bash
cd backend && uv sync
```

```bash
uv run --project backend python -c "from mini_vllm.model import load; load('cpu')"
```

The second command downloads the pinned model revision (`c1899de`, ~1.5 GB) into the Hugging Face cache.

```bash
cd frontend && npm install && npm run build
```

## Run

```bash
cd backend && HF_HUB_OFFLINE=1 uv run uvicorn mini_vllm.server:app --workers 1 --port 8077
```

Open http://127.0.0.1:8077 (the backend serves `frontend/dist`). Always use **one** worker: each worker would load its
own engine. Set `MINI_VLLM_DEVICE=cpu` to force CPU.

For UI development run `npm run dev` in `frontend/` (proxies `/api` to port 8077, override with `MINI_VLLM_PORT`).
`?fixture=batching` or `?fixture=single` replays the illustrative contract traces with no backend, clearly labelled.

## Defaults and limits (R0)

| Setting | Value |
| --- | --- |
| Block size | 16 tokens (fixed per run) |
| KV pool | 256 blocks = 448 MiB bf16, allocated at startup |
| Active requests | 8; waiting queue 64 (then 429) |
| Request budget | prompt + max_new_tokens ≤ 2048; max_new_tokens ≤ 256 (default 128) |
| Admission | FIFO, conservative credits `ceil((prompt + max_new) / block_size)` |
| Sampling | greedy by default; temperature + seed supported |
| Journal | SQLite WAL at `backend/data/journal.sqlite`; keeps 20 finished runs / 200 MiB |

Known limitations: one bounded prefill between decode steps (no chunked prefill); gather-based attention (extra
copies, no paged kernel); no prefix sharing or preemption (R1); GPU duration is reported as unavailable (host
timings only); MPS kernels compile on first use of a new shape and run slowly after the GPU has been idle, so the
first request after startup or a long pause has inflated latency.

## Tests

```bash
cd backend && HF_HUB_OFFLINE=1 uv run pytest
```

```bash
cd frontend && npm test
```

The backend suite loads the model on CPU (~90 s). `-m "not model"` runs only the fast allocator/fixture tests.

## Benchmarks

```bash
HF_HUB_OFFLINE=1 uv run --project backend python benchmarks/run.py --quick
```

Writes `benchmarks/results/<timestamp>/{manifest.json,raw.jsonl,summary.json,report.md}`. Small samples are
labelled exploratory.
