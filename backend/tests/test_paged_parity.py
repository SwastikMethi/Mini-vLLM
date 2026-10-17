"""Paged runner vs stock HF forward on CPU fp32: single requests across block boundaries, then mixed batches."""

import pytest
import torch

from mini_vllm.kv_cache import KVPool
from mini_vllm.runner import DecodeRow, Runner

pytestmark = pytest.mark.model
ATOL = 2e-3  # fp32 logits; measured max deviation is recorded in docs/PARITY.md
STEPS = 8


def make_runner(model, blocks=64, bs=16):
    cfg = model.config
    pool = KVPool(layers=cfg.num_hidden_layers, blocks=blocks, block_size=bs, kv_heads=cfg.num_key_value_heads,
                  head_dim=cfg.head_dim, dtype=torch.float32, device="cpu")
    return Runner(model, pool), pool


@torch.inference_mode()
def hf_logits(model, ids):
    return model(torch.tensor([ids])).logits[0].float()


def paged_greedy(runner, pool, rid, ids, steps):
    pool.reserve(rid, pool.blocks_for(len(ids) + steps))
    pool.ensure(rid, len(ids))
    out = [runner.prefill(rid, ids)]
    toks = [int(out[-1].argmax())]
    for _ in range(steps - 1):
        pos = len(ids) + len(toks) - 1
        pool.ensure(rid, pos + 1)
        out.append(runner.decode([DecodeRow(rid, toks[-1], pos)])[0])
        toks.append(int(out[-1].argmax()))
    return toks, torch.stack(out)


@pytest.mark.parametrize("bs", [16, 4])
@pytest.mark.parametrize("n", ["15", "16", "17", "31", "32", "33", "65"])
def test_single_request_matches_reference(cpu_model, reference_cpu, n, bs):
    model, _ = cpu_model
    case = reference_cpu[n]
    runner, pool = make_runner(model, bs=bs, blocks=64 if bs == 16 else 256)
    toks, logits = paged_greedy(runner, pool, "A", case["prompt_ids"], STEPS)
    assert toks == case["greedy"]
    ref = hf_logits(model, case["prompt_ids"] + toks[:-1])[len(case["prompt_ids"]) - 1 :]
    assert (logits - ref).abs().max() < ATOL


def test_mixed_batch_matches_isolated_and_survives_membership_change(cpu_model, reference_cpu):
    """A, B, C at different lengths decode together; B leaves early, C joins late. Each row must equal its
    isolated reference: no cross-request reads, correct per-row positions after membership changes."""
    model, _ = cpu_model
    runner, pool = make_runner(model)
    prompts = {"A": reference_cpu["33"]["prompt_ids"], "B": reference_cpu["15"]["prompt_ids"],
               "C": reference_cpu["65"]["prompt_ids"]}
    expected = {r: reference_cpu[k]["greedy"] for r, k in (("A", "33"), ("B", "15"), ("C", "65"))}
    toks, logits = {}, {}

    def admit(rid):
        ids = prompts[rid]
        pool.reserve(rid, pool.blocks_for(len(ids) + STEPS))
        pool.ensure(rid, len(ids))
        lg = runner.prefill(rid, ids)
        logits[rid], toks[rid] = [lg], [int(lg.argmax())]

    admit("A"), admit("B")
    for step in range(STEPS + 3):
        if step == 2:
            admit("C")  # newcomer prefills between decode steps
        active = [r for r in ("A", "B", "C") if r in toks and len(toks[r]) < (4 if r == "B" else STEPS)]
        if not active:
            break
        rows = []
        for r in active:
            pos = len(prompts[r]) + len(toks[r]) - 1
            pool.ensure(r, pos + 1)
            rows.append(DecodeRow(r, toks[r][-1], pos))
        out = runner.decode(rows)
        for i, r in enumerate(active):
            logits[r].append(out[i])
            toks[r].append(int(out[i].argmax()))
        if "B" in toks and len(toks["B"]) == 4 and "B" in pool.tables:
            pool.release("B")  # dynamic removal; its blocks may be reused by C's growth

    for r in ("A", "B", "C"):
        n = len(toks[r])
        assert toks[r] == expected[r][:n], r
        ref = hf_logits(model, prompts[r] + toks[r][:-1])[len(prompts[r]) - 1 :]
        assert (torch.stack(logits[r]) - ref).abs().max() < ATOL, r


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_bf16_within_tolerance(reference_cpu):
    from mini_vllm.model import load

    model, _ = load("mps")
    cfg = model.config
    for n in ("16", "33", "65"):  # the 15-token case is a documented bf16 near-tie (docs/PARITY.md)
        case = reference_cpu[n]
        pool = KVPool(layers=cfg.num_hidden_layers, blocks=64, block_size=16, kv_heads=cfg.num_key_value_heads,
                      head_dim=cfg.head_dim, dtype=torch.bfloat16, device="mps")
        toks, logits = paged_greedy(Runner(model, pool), pool, "A", case["prompt_ids"], STEPS)
        assert toks == case["greedy"], n
        with torch.inference_mode():
            ref = model(torch.tensor([case["prompt_ids"] + toks[:-1]], device="mps")).logits[0].float()
        assert (logits - ref[len(case["prompt_ids"]) - 1 :]).abs().max() < 0.5, n
