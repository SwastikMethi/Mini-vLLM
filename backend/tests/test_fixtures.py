"""The reducer must reproduce the checked-in snapshots from the checked-in traces."""

import json

import jsonschema
import pytest

from mini_vllm.config import CONTRACTS_DIR
from mini_vllm.snapshot import Projection, SequenceGap, replay, request_metrics

SCHEMA = json.loads((CONTRACTS_DIR / "event_schema.json").read_text())
CASES = [
    ("continuous_batching_example.jsonl", "continuous_batching_checkpoint.json"),
    ("continuous_batching_example.jsonl", "continuous_batching_snapshot.json"),
    ("event_examples.jsonl", "snapshot_example.json"),
]


def load_events(name):
    return [json.loads(line) for line in (CONTRACTS_DIR / name).read_text().splitlines() if line.strip()]


def assert_matches(snap, expected):
    """Compare every field the fixture states; the reducer may carry extra fields (e.g. prompt)."""
    assert snap["high_watermark_seq"] == expected["high_watermark_seq"]
    assert snap["cache"] == expected["cache"]
    assert [r["request_id"] for r in snap["requests"]] == [r["request_id"] for r in expected["requests"]]
    for got, want in zip(snap["requests"], expected["requests"]):
        assert {k: got[k] for k in want} == want, want["request_id"]
    assert snap["batches"] == expected["batches"]


@pytest.mark.parametrize("trace", sorted({t for t, _ in CASES}))
def test_fixture_events_validate(trace):
    for e in load_events(trace):
        jsonschema.validate(e, SCHEMA)


@pytest.mark.parametrize("trace,snapshot", CASES)
def test_replay_reproduces_snapshot(trace, snapshot):
    expected = json.loads((CONTRACTS_DIR / snapshot).read_text())
    events = [e for e in load_events(trace) if e["seq"] <= expected["high_watermark_seq"]]
    assert_matches(replay(events, fixture=True).snapshot(), expected)


def test_request_metrics_match_example():
    expected = json.loads((CONTRACTS_DIR / "snapshot_example.json").read_text())["metrics"]
    req = replay(load_events("event_examples.jsonl")).snapshot()["requests"][0]
    assert request_metrics(req) == expected


def test_duplicates_ignored_and_gaps_detected():
    events = load_events("continuous_batching_example.jsonl")
    p = Projection()
    for e in events[:10]:
        p.apply(e)
    assert p.apply(events[5]) is False
    with pytest.raises(SequenceGap):
        p.apply(events[11])


def test_queue_update_keeps_first_entry_time_and_sampling_adds_no_slot():
    p = replay(load_events("continuous_batching_example.jsonl"))
    c = p.requests["C"]
    assert c["queued_ns"] == 69_000_000  # re-queued at 86 ms with a new reason; first entry is kept
    p = replay(load_events("continuous_batching_example.jsonl")[:13])  # through A's first token.sampled
    assert p.requests["A"]["generated_tokens"] == 1 and p.requests["A"]["cached_tokens"] == 15
