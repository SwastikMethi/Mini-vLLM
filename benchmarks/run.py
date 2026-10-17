"""Benchmark: HF one-at-a-time reference vs HF static batching vs the continuous paged engine.

Every mode sees the same seeded workload (prompts, caps, arrival schedule), greedy sampling, EOS + cap stop policy,
on the same device/dtype/revision. Latencies are measured from each request's scheduled arrival on one host
monotonic clock. A warm-up pass of the identical workload runs first (MPS compiles kernels per new shape).

  uv run --project backend python benchmarks/run.py --quick
  uv run --project backend python benchmarks/run.py --concurrency 1 2 4 8 --prompt-tokens 32 256 1024 \
      --max-new 32 128 --arrivals burst paced --requests 1000

Conditions with fewer than 1000 completions are labelled exploratory; p99 from them is not a reported result.
"""

import argparse
import json
import random
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from mini_vllm.config import Config, default_device
from mini_vllm.engine import Engine
from mini_vllm.model import encode_chat, load, manifest

RESULTS = Path(__file__).resolve().parent / "results"
FILLER = ("memory pages blocks tokens cache scheduler batch decode prefill attention request queue latency "
          "throughput kernel tensor layer head model server engine").split()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_workload(tok, n: int, prompt_tokens: int, max_new: int, arrivals: str, rate: float, seed: int) -> list[dict]:
    """Seeded prompts sized so the chat-templated prompt is ~prompt_tokens, plus an arrival offset per request."""
    rng = random.Random(seed)
    overhead = len(encode_chat(tok, ""))
    work, t = [], 0.0
    for i in range(n):
        words = [rng.choice(FILLER) for _ in range(max(1, prompt_tokens * 2))]
        ids = tok(f"Request {i}. Summarize: " + " ".join(words), add_special_tokens=False)["input_ids"]
        text = tok.decode(ids[: max(1, prompt_tokens - overhead)])
        if arrivals == "paced":
            t += rng.expovariate(rate)
        work.append({"i": i, "prompt": text, "max_new": max_new, "arrival_s": 0.0 if arrivals == "burst" else t})
    return work


class StepClock(LogitsProcessor):
    """Records a device-synchronized timestamp each time generate produces next-token scores (one per step)."""

    def __init__(self, device):
        self.device, self.times = device, []

    def __call__(self, input_ids, scores):
        sync(self.device)
        self.times.append(time.perf_counter())
        return scores


@torch.inference_mode()
def hf_generate(model, tok, prompts: list[str], max_new: int, device: str):
    tok.padding_side = "left"
    texts = [tok.decode(encode_chat(tok, p)) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    clock = StepClock(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                         logits_processor=LogitsProcessorList([clock]), pad_token_id=tok.pad_token_id)
    sync(device)
    end = time.perf_counter()
    eos = set(model.generation_config.eos_token_id)
    rows = []
    for r in out[:, enc["input_ids"].shape[1]:].tolist():
        n = next((k + 1 for k, t in enumerate(r) if t in eos), len(r))
        rows.append({"output_tokens": n, "prompt_tokens": int(enc["attention_mask"][len(rows)].sum())})
    return rows, clock.times, end


def run_hf(model, tok, work, concurrency: int, device: str, static: bool) -> list[dict]:
    """reference: one request at a time (concurrency ignored). static: groups of `concurrency` in arrival order;
    a group starts when its last member has arrived and the previous group finished; all members return together."""
    group = concurrency if static else 1
    t0, free_at, results = time.perf_counter(), 0.0, []
    for g in range(0, len(work), group):
        members = work[g: g + group]
        start = max(free_at, max(w["arrival_s"] for w in members))
        while time.perf_counter() - t0 < start:
            time.sleep(0.001)
        rows, steps, end = hf_generate(model, tok, [w["prompt"] for w in members], members[0]["max_new"], device)
        free_at = end - t0
        for w, row in zip(members, rows):
            arrival = t0 + w["arrival_s"]
            tok_times = steps[: row["output_tokens"]]
            results.append({**row, "i": w["i"], "status": "completed",
                            "ttft_ms": (tok_times[0] - arrival) * 1e3,
                            "e2e_ms": ((end if static else tok_times[-1]) - arrival) * 1e3,
                            "itl_ms": [(b - a) * 1e3 for a, b in zip(tok_times, tok_times[1:])],
                            "done_s": (end if static else tok_times[-1]) - t0})
    return results


def run_engine(model, tok, work, concurrency: int, device: str, pool_blocks: int) -> list[dict]:
    events, done = [], threading.Condition()

    def sink(group):
        with done:
            events.extend(group)
            done.notify_all()

    eng = Engine(Config(device=device, max_active=concurrency, pool_blocks=pool_blocks, deadline_ms=600_000),
                 model, tok, sink)
    eng.start()
    with done:
        done.wait_for(lambda: any(e["type"] == "engine.ready" for e in events), 300)
    arrivals, t0 = {}, time.perf_counter()
    for w in work:
        while time.perf_counter() - t0 < w["arrival_s"]:
            time.sleep(0.0005)
        ns = eng.log.now()
        rid = eng.submit({"prompt": w["prompt"], "max_new_tokens": w["max_new"], "temperature": 0.0, "seed": 0,
                          "client_request_id": f"bench-{w['i']}"}).result()
        arrivals[rid] = (w["i"], ns)
    with done:
        done.wait_for(lambda: sum(e["type"] == "request.terminal" for e in events) == len(work), 3600)
    snap = eng.log.snapshot()
    eng.stop()
    base = eng.log.now() - (time.perf_counter() - t0) * 1e9  # engine-clock value at t0
    results = []
    for r in snap["requests"]:
        i, arrival = arrivals[r["request_id"]]
        s = r["sampled_ns"]
        results.append({"i": i, "status": r["status"], "prompt_tokens": r["prompt_tokens"],
                        "output_tokens": r["generated_tokens"], "ttft_ms": (s[0] - arrival) / 1e6 if s else None,
                        "e2e_ms": (r["terminal_ns"] - arrival) / 1e6,
                        "itl_ms": [(b - a) / 1e6 for a, b in zip(s, s[1:])],
                        "done_s": (r["terminal_ns"] - base) / 1e9})
    results.append({"_batches": [(len(b["members"]), b["phase"], b["host_duration_ms"]) for b in snap["batches"]]})
    return results


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, round(q / 100 * (len(xs) - 1)))]


def summarize(rows: list[dict], work) -> dict:
    ok = [r for r in rows if r.get("status") == "completed"]
    span = max(r["done_s"] for r in ok) - min(w["arrival_s"] for w in work) if ok else None
    e2e, ttft = [r["e2e_ms"] for r in ok], [r["ttft_ms"] for r in ok]
    itl = [x for r in ok for x in r["itl_ms"]]
    out_tokens = sum(r["output_tokens"] for r in ok)
    return {
        "completed": len(ok), "failed": len(rows) - len(ok),
        "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95)},
        "itl_ms": {"p50": pct(itl, 50), "p95": pct(itl, 95)},
        "e2e_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95), "p99": pct(e2e, 99)},
        "output_tokens": out_tokens, "output_tok_per_s": out_tokens / span if span else None,
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in ok) if ok else None,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in ok) if ok else None,
        "exploratory": len(ok) < 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[32, 256, 1024])
    ap.add_argument("--max-new", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--arrivals", nargs="+", default=["burst", "paced"], choices=["burst", "paced"])
    ap.add_argument("--rate", type=float, default=4.0, help="paced arrivals per second (Poisson)")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--modes", nargs="+", default=["reference", "static", "continuous"])
    ap.add_argument("--pool-blocks", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small exploratory matrix")
    a = ap.parse_args()
    if a.quick:
        a.concurrency, a.prompt_tokens, a.max_new, a.arrivals, a.requests = [1, 4], [32, 256], [32], ["burst"], 8

    model, tok = load(a.device)
    out = RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    man = {**manifest(model, a.device), "block_size": 16, "pool_blocks": a.pool_blocks, "trace_mode": "engine-events-on",
           "measurement": "host perf_counter / engine monotonic clock, device-synchronized; from scheduled arrival",
           "stop_policy": "eos or max_new_tokens", "sampling": "greedy", "args": vars(a)}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    summary, raw = [], (out / "raw.jsonl").open("w")
    for c in a.concurrency:
        for p in a.prompt_tokens:
            for m in a.max_new:
                for arr in a.arrivals:
                    work = make_workload(tok, a.requests, p, m, arr, a.rate, a.seed)
                    for mode in a.modes:
                        if mode == "static" and c == 1:
                            continue  # identical to reference
                        go = (lambda: run_engine(model, tok, work, c, a.device, a.pool_blocks)) if mode == "continuous" \
                            else (lambda: run_hf(model, tok, work, c, a.device, static=mode == "static"))
                        go()  # warm-up on the identical workload; discarded
                        rows = go()
                        extra = rows.pop() if rows and "_batches" in rows[-1] else None
                        cond = {"mode": mode, "concurrency": c, "prompt_tokens": p, "max_new": m, "arrivals": arr}
                        for r in rows:
                            raw.write(json.dumps({**cond, **r}) + "\n")
                        s = {**cond, **summarize(rows, work)}
                        if extra:
                            sizes = [n for n, ph, _ in extra["_batches"] if ph == "decode"]
                            s["mean_decode_batch"] = statistics.mean(sizes) if sizes else None
                        summary.append(s)
                        print(json.dumps({k: s[k] for k in ("mode", "concurrency", "prompt_tokens", "max_new", "arrivals",
                                                            "completed", "output_tok_per_s")}
                                         | {"ttft_p50": s["ttft_ms"]["p50"], "e2e_p95": s["e2e_ms"]["p95"]}), flush=True)
    raw.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out, man, summary)
    print(f"results in {out}")


def write_report(out: Path, man: dict, summary: list[dict]) -> None:
    f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [f"# Benchmark report — {out.name}", "",
             f"Device **{man['device']}** ({man['device_name']}), dtype {man['dtype']}, torch {man['torch']}, "
             f"transformers {man['transformers']}, model {man['model_id']}@{man['model_revision'][:7]}.",
             f"Measurement: {man['measurement']}. Trace mode: {man['trace_mode']}. Stop policy: {man['stop_policy']}.",
             "", "**Exploratory**: fewer than 1000 completions per condition — p99 and small differences are not "
             "reportable results." if any(s["exploratory"] for s in summary) else "", "",
             "| mode | conc | prompt | max_new | arrivals | done/failed | out tok/s | TTFT p50 ms | TTFT p95 ms | "
             "ITL p50 ms | E2E p50 ms | E2E p95 ms | mean decode batch |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['mode']} | {s['concurrency']} | {s['prompt_tokens']} | {s['max_new']} | {s['arrivals']} | "
                     f"{s['completed']}/{s['failed']} | {f(s['output_tok_per_s'])} | {f(s['ttft_ms']['p50'])} | "
                     f"{f(s['ttft_ms']['p95'])} | {f(s['itl_ms']['p50'])} | {f(s['e2e_ms']['p50'])} | "
                     f"{f(s['e2e_ms']['p95'])} | {f(s.get('mean_decode_batch'), 2)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
