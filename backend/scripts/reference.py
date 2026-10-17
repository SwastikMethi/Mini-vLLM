"""M0 reference harness: greedy tokens + top-5 logits per step from the stock HF forward.

Usage: uv run python scripts/reference.py [cpu] [mps]
Writes tests/data/reference_<device>.json and prints CPU-vs-MPS greedy agreement.
"""

import json
import sys
from pathlib import Path

import torch

from mini_vllm.model import encode_chat, load, manifest

PROMPT = (
    "Explain, step by step and in plain language, how a paged key-value cache lets a language model "
    "server keep many conversations in GPU memory at once without reserving one long contiguous buffer "
    "per request, and why fixed-size blocks make freed memory easy to reuse for newly arriving requests."
)
LENGTHS = [15, 16, 17, 31, 32, 33, 65]
DECODE_STEPS = 8
OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


@torch.inference_mode()
def reference(model, ids: list[int], steps: int) -> dict:
    out = model.generate(
        torch.tensor([ids], device=model.device), max_new_tokens=steps, do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )
    full = out[0].tolist()
    # Re-run one full forward so every step's logits come from the same uncached computation.
    logits = model(torch.tensor([full[:-1]], device=model.device)).logits[0, len(ids) - 1 :].float().cpu()
    top = logits.topk(5, dim=-1)
    return {
        "prompt_ids": ids,
        "greedy": full[len(ids) :],
        "top5_ids": top.indices.tolist(),
        "top5_logits": top.values.tolist(),
    }


def main(devices: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for device in devices:
        model, tok = load(device)
        base = encode_chat(tok, PROMPT)
        assert len(base) >= max(LENGTHS), len(base)
        cases = {str(n): reference(model, base[:n], DECODE_STEPS) for n in LENGTHS}
        results[device] = cases
        path = OUT / f"reference_{device}.json"
        path.write_text(json.dumps({"manifest": manifest(model, device), "cases": cases}))
        print(f"wrote {path}")
        del model
    if {"cpu", "mps"} <= results.keys():
        for n in LENGTHS:
            a, b = results["cpu"][str(n)]["greedy"], results["mps"][str(n)]["greedy"]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            print(f"len {n:>3}: cpu==mps greedy {'yes' if first is None else f'NO, diverges at step {first}'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cpu", "mps"])
