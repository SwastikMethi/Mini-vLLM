"""Engine scheduling and lifecycle with the real model on CPU and a deliberately small pool."""

import json
import threading
import time

import jsonschema
import pytest

from mini_vllm.config import CONTRACTS_DIR, Config
from mini_vllm.engine import Engine, EngineError
from mini_vllm.snapshot import replay

pytestmark = pytest.mark.model
SCHEMA = json.loads((CONTRACTS_DIR / "event_schema.json").read_text())


class Collector:
    def __init__(self):
        self.events, self.cv = [], threading.Condition()

    def __call__(self, group):
        with self.cv:
            self.events.extend(group)
            self.cv.notify_all()

    def wait(self, pred, timeout=120):
        with self.cv:
            assert self.cv.wait_for(lambda: pred(self.events), timeout), "timed out waiting for engine events"

    def of(self, type_, rid=None):
        return [e for e in self.events if e["type"] == type_ and (rid is None or e["request_id"] == rid)]


@pytest.fixture
def engine(cpu_model):
    def make(**cfg):
        model, tok = cpu_model
        sink = Collector()
        eng = Engine(Config(device="cpu", **cfg), model, tok, sink)
        eng.start()
        sink.wait(lambda ev: any(e["type"] == "engine.ready" for e in ev))
        made.append(eng)
        return eng, sink

    made = []
    yield make
    for eng in made:
        eng.stop()


def body(prompt, n=8, cid=None, **kw):
    return {"prompt": prompt, "max_new_tokens": n, "temperature": 0.0, "seed": 0,
            "client_request_id": cid or prompt, **kw}


def terminal(sink, rid):
    return lambda ev: any(e["type"] == "request.terminal" and e["request_id"] == rid for e in ev)


def assert_clean(eng, sink):
    assert eng.pool.free_blocks == eng.pool.num_blocks and eng.pool.credits == {} and eng.pool.owner == {}
    for e in sink.events:
        jsonschema.validate(e, SCHEMA)
    assert [e["seq"] for e in sink.events] == list(range(1, len(sink.events) + 1))
    assert replay(sink.events, manifest=eng.manifest).snapshot() == eng.log.snapshot()  # the trace reproduces the live state


def test_continuous_batching_newcomer_shared_batch_and_capacity_wait(engine):
    eng, sink = engine(pool_blocks=6)
    a = eng.submit(body("Count from one to twenty.", n=24)).result()  # ~44 tokens: 3 credits
    sink.wait(lambda ev: any(e["type"] == "request.decode.completed" and e["request_id"] == a for e in ev))
    b = eng.submit(body("Name a color.", n=20)).result()  # 3 credits → 6 of 6 reserved
    c = eng.submit(body("Name a fruit.", n=20)).result()  # waits for capacity
    for rid in (a, b, c):
        sink.wait(terminal(sink, rid))

    batches = {e["batch_id"]: e["payload"] for e in sink.of("batch.scheduled")}
    b_prefill = next(bid for bid, p in batches.items() if p["phase"] == "prefill" and p["members"][0]["request_id"] == b)
    a_decodes = [e["batch_id"] for e in sink.of("request.decode.started", a)]
    assert a_decodes[0] < b_prefill < a_decodes[-1], "B prefilled between A's decode steps"
    shared = [bid for bid, p in batches.items() if {m["request_id"] for m in p["members"]} >= {a, b}]
    assert shared, "A and B shared at least one decode batch"
    assert sink.of("request.queued", c)[0]["payload"]["reason"] == "capacity"
    c_admit = sink.of("request.admitted", c)[0]["seq"]
    assert c_admit > min(e["seq"] for e in sink.of("cache.released"))  # admitted only after a release
    # Completed forward intervals in the single engine lane never overlap.
    spans = sorted((e["monotonic_ns"], e["batch_id"]) for e in sink.of("batch.started"))
    ends = {e["batch_id"]: e["monotonic_ns"] for e in sink.of("batch.completed")}
    for (_, x), (start_y, _) in zip(spans, spans[1:]):
        assert ends[x] <= start_y
    assert_clean(eng, sink)


def test_token_events_and_slot_commits(engine):
    eng, sink = engine()
    rid = eng.submit(body("Say hello.", n=5)).result()
    sink.wait(terminal(sink, rid))
    toks = sink.of("token.sampled", rid)
    assert [e["payload"]["token_index"] for e in toks] == list(range(len(toks)))
    term = sink.of("request.terminal", rid)[0]["payload"]
    assert "".join(e["payload"]["text_delta"] for e in toks) == term["final_output_text"] or term["stop_reason"] == "eos"
    p = sink.of("request.tokenized", rid)[0]["payload"]["prompt_tokens"]
    # The last sampled token is never cached: processed = prompt + decode steps = prompt + generated - 1.
    assert term["processed_tokens"] == p + term["generated_tokens"] - 1
    assert_clean(eng, sink)


def test_limit_one_finishes_after_prefill(engine):
    eng, sink = engine()
    rid = eng.submit(body("Hi", n=1)).result()
    sink.wait(terminal(sink, rid))
    assert not sink.of("request.decode.started", rid)
    assert sink.of("request.terminal", rid)[0]["payload"]["stop_reason"] in ("length", "eos")
    assert_clean(eng, sink)


def test_cancel_queued_and_active(engine):
    eng, sink = engine(pool_blocks=6)
    a = eng.submit(body("Write a long story about a lighthouse.", n=60)).result()  # ~82 tokens: whole pool
    b = eng.submit(body("Queued then cancelled.", n=8)).result()
    sink.wait(lambda ev: any(e["type"] == "request.decode.completed" and e["request_id"] == a for e in ev))
    assert eng.cancel(b).result() == "cancel_requested"
    assert eng.cancel(a).result() == "cancel_requested"
    for rid in (a, b):
        sink.wait(terminal(sink, rid))
    assert eng.cancel(a).result() == "cancelled"  # idempotent once terminal
    types = [e["type"] for e in sink.events if e["request_id"] == a]
    assert types.index("request.cancel_requested") < types.index("cache.released") < types.index("request.terminal")
    assert not sink.of("cache.released", b)  # never owned anything
    assert sink.of("request.terminal", b)[0]["payload"]["status"] == "cancelled"
    with pytest.raises(EngineError) as err:
        eng.cancel("RQ-999").result()
    assert err.value.http_status == 404
    assert_clean(eng, sink)


def test_admission_errors_and_idempotency(engine):
    eng, sink = engine(pool_blocks=2)
    first = eng.submit(body("Same body", cid="k1")).result()
    assert eng.submit(body("Same body", cid="k1")).result() == first
    with pytest.raises(EngineError) as err:
        eng.submit(body("Different body", cid="k1")).result()
    assert err.value.http_status == 409
    with pytest.raises(EngineError) as err:
        eng.submit(body("Too big for a two-block pool", n=200)).result()
    assert err.value.http_status == 422
    sink.wait(terminal(sink, first))
    assert_clean(eng, sink)


def test_forward_failure_fails_run_and_releases(engine, monkeypatch):
    eng, sink = engine()
    def boom(rows):
        raise RuntimeError("MPS backend out of memory (simulated)")
    monkeypatch.setattr(eng.runner, "decode", boom)
    rid = eng.submit(body("This decode will fail.", n=8)).result()
    sink.wait(lambda ev: any(e["type"] == "engine.failed" for e in ev))
    assert sink.of("batch.failed")[0]["payload"]["error_code"] == "device_oom"
    t = sink.of("request.terminal", rid)[0]["payload"]
    assert (t["status"], t["stop_reason"]) == ("failed", "engine_error")
    with pytest.raises(EngineError) as err:
        eng.submit(body("after failure")).result(timeout=5)
    assert err.value.http_status == 503
    assert_clean(eng, sink)


def test_deadline_cancels(engine):
    eng, sink = engine(deadline_ms=1)
    rid = eng.submit(body("Deadline", n=50)).result()
    time.sleep(0.01)
    sink.wait(terminal(sink, rid))
    assert sink.of("request.cancel_requested", rid)[0]["payload"]["reason"] == "deadline"
    assert sink.of("request.terminal", rid)[0]["payload"]["stop_reason"] == "deadline"
    assert_clean(eng, sink)
