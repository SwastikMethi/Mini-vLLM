"""HTTP + SSE contract against a real Uvicorn server (CPU model, temp journal)."""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from mini_vllm import server as srv
from mini_vllm.config import Config
from mini_vllm.snapshot import replay

pytestmark = pytest.mark.model


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Live:
    def __init__(self, cpu_model, data_dir, **cfg):
        self.app = srv.create_app(Config(device="cpu", data_dir=data_dir, **cfg), loader=lambda device: cpu_model)
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(self.app, port=port, log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.http = httpx.Client(base_url=self.url, timeout=60)
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self.http.get("/api/health").json()["status"] == "ready":
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        self.run_id = self.http.get("/api/health").json()["run_id"]

    @property
    def st(self):
        return self.app.state.st

    def submit(self, prompt, n=4, cid=None, **kw):
        return self.http.post("/api/requests", json={"prompt": prompt, "max_new_tokens": n,
                                                      "client_request_id": cid or prompt, **kw})

    def read_sse(self, until, headers=None, params=None, timeout=60):
        """Collect (event, id, data) records until until(records) is true or the server closes the stream."""
        records, cur = [], {}
        with self.http.stream("GET", f"/api/runs/{self.run_id}/events", headers=headers, params=params,
                              timeout=timeout) as r:
            for line in r.iter_lines():
                if line.startswith(":"):
                    continue
                if not line:
                    if cur:
                        records.append(cur)
                        cur = {}
                        if until(records):
                            break
                    continue
                k, _, v = line.partition(": ")
                cur[k] = json.loads(v) if k == "data" else v
        return records

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=30)


@pytest.fixture
def live(cpu_model, tmp_path):
    made = []

    def make(**cfg):
        made.append(Live(cpu_model, tmp_path, **cfg))
        return made[-1]

    yield make
    for x in made:
        x.close()


def terminal_for(rid):
    return lambda recs: any(r["data"]["type"] == "request.terminal" and r["data"]["request_id"] == rid for r in recs)


def test_submit_contract(live):
    s = live()
    r = s.submit("Hello there", cid="k1")
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    rid = r.json()["request_id"]
    assert s.submit("Hello there", cid="k1").json()["request_id"] == rid  # identical retry
    assert s.submit("Changed", cid="k1").status_code == 409
    bad = s.http.post("/api/requests", json={"prompt": "", "client_request_id": "x"})
    assert bad.status_code == 422 and set(bad.json()["error"]) == {"code", "message", "retryable"}
    assert s.http.post("/api/requests", json={"prompt": "x", "client_request_id": "y", "extra": 1}).status_code == 422
    assert s.http.post("/api/requests/RQ-999/cancel").status_code == 404
    s.read_sse(terminal_for(rid))
    assert s.http.post(f"/api/requests/{rid}/cancel").json() == {"request_id": rid, "status": "completed"}
    assert [x["run_id"] for x in s.http.get("/api/runs").json()] == [s.run_id]


def test_stream_resume_snapshot_and_export(live):
    s = live()
    rid = s.submit("Tell me a fact about owls.", n=6).json()["request_id"]
    full = s.read_sse(terminal_for(rid))
    assert all(r["event"] == "engine_event" and r["id"] == f"{s.run_id}:{r['data']['seq']}" for r in full)
    seqs = [r["data"]["seq"] for r in full]
    assert seqs == list(range(1, len(seqs) + 1))

    # Reconnect mid-trace with Last-Event-ID: only later events, no duplicates.
    mid = seqs[len(seqs) // 2]
    tail = s.read_sse(terminal_for(rid), headers={"Last-Event-ID": f"{s.run_id}:{mid}"})
    assert [r["data"]["seq"] for r in tail] == seqs[mid:]
    # Last-Event-ID takes precedence over after_seq.
    pref = s.read_sse(lambda recs: True, headers={"Last-Event-ID": f"{s.run_id}:{mid}"}, params={"after_seq": 0})
    assert pref[0]["data"]["seq"] == mid + 1

    # Snapshot then subscribe after its watermark reproduces the same state as the full stream.
    snap = s.http.get(f"/api/runs/{s.run_id}/snapshot").json()
    assert snap["high_watermark_seq"] == seqs[-1] and snap["engine_status"] == "ready"
    rid2 = s.submit("And one about bats.", n=3).json()["request_id"]
    after = s.read_sse(terminal_for(rid2), params={"after_seq": snap["high_watermark_seq"]})
    assert after[0]["data"]["seq"] == snap["high_watermark_seq"] + 1
    everything = [r["data"] for r in full] + [r["data"] for r in after]
    assert replay(everything, manifest=snap["manifest"]).snapshot() == s.http.get(f"/api/runs/{s.run_id}/snapshot").json()

    exp = s.http.get(f"/api/runs/{s.run_id}/export").json()
    assert [e["seq"] for e in exp["events"]] == list(range(1, len(everything) + 1))
    assert exp["initial_state"]["high_watermark_seq"] == 0
    assert exp["manifest"]["device"] == "cpu" and exp["manifest"]["trace_mode"] == "live"


def test_stale_cursor_resets(live):
    s = live()
    for headers, params in [({"Last-Event-ID": "other-run:3"}, None), (None, {"after_seq": 10_000})]:
        recs = s.read_sse(lambda r: True, headers=headers, params=params)
        assert recs[0]["event"] == "reset"
        assert recs[0]["data"]["snapshot_url"] == f"/api/runs/{s.run_id}/snapshot"
    assert s.http.get("/api/runs/nope/events").status_code == 404
    assert s.http.get("/api/runs/nope/snapshot").status_code == 404


def test_slow_subscriber_is_dropped_not_waited_for():
    async def go():
        b = srv.Broker(asyncio.get_running_loop())
        sub = b.subscribe()
        sub.queue = asyncio.Queue(1)
        b._fanout([{"seq": 1}])
        b._fanout([{"seq": 2}])  # queue full: drop the subscriber, engine never blocks
        assert sub.overflow and sub not in b.subs

    asyncio.run(go())


def test_journal_failure_fails_run_visibly(live):
    s = live()
    def broken(group):
        raise OSError("disk full (simulated)")
    s.st.journal.append = broken
    with s.http.stream("GET", f"/api/runs/{s.run_id}/events", params={"after_seq": 1}, timeout=60) as r:
        s.submit("This cannot be journaled")
        for line in r.iter_lines():
            if line.startswith("data: "):
                e = json.loads(line[6:])
                if e["type"] == "engine.failed":
                    assert e["payload"]["trace_complete"] is False
                    break
    assert s.http.get("/api/health").json()["status"] == "failed"
    assert s.submit("after failure", cid="z").status_code == 503
    assert s.st.engine.pool.credits == {} and s.st.engine.pool.owner == {}


def test_restart_creates_new_run_and_old_run_replays(live, cpu_model, tmp_path):
    s = live()
    rid = s.submit("First run request", n=3).json()["request_id"]
    s.read_sse(terminal_for(rid))
    old_run, old_snap = s.run_id, s.http.get(f"/api/runs/{s.run_id}/snapshot").json()
    s.close()
    time.sleep(1.1)  # run ids have one-second resolution
    s2 = live()
    assert s2.run_id != old_run
    runs = {x["run_id"]: x for x in s2.http.get("/api/runs").json()}
    assert runs[old_run]["status"] == "completed"
    assert s2.http.get(f"/api/runs/{old_run}/snapshot").json() == old_snap
    assert s2.submit("First run request", n=3).status_code == 202  # idempotency is scoped to a run
