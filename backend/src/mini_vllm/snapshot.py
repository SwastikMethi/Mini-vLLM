"""Deterministic event reducer → snapshot. The engine folds every emitted event through this, so /snapshot is
exactly the journal's state; replaying a recorded trace through it must reproduce the same snapshot.

Projection rules follow contracts/ui_projection_contract.md.
"""

import copy


class SequenceGap(RuntimeError):
    pass


def _new_request(rid: str, ns: int, p: dict) -> dict:
    return {
        "request_id": rid, "status": "accepted", "stop_reason": None, "prompt": p["prompt"],
        "prompt_tokens": 0, "generated_tokens": 0, "processed_tokens": 0, "cached_tokens": 0,
        "token_ids": [], "output_text": "", "block_table": [], "reservation_blocks": 0, "decode_step": 0,
        "cancel_requested": False, "accepted_ns": ns, "queued_ns": None, "admitted_ns": None,
        "prefill_started_ns": None, "terminal_ns": None, "wait_reason": None, "queue_position": None,
        "executing_batch_id": None, "sampled_ns": [], "max_new_tokens": p["max_new_tokens"], "error_code": None,
    }


class Projection:
    def __init__(self, run_id: str | None = None, manifest: dict | None = None, fixture: bool = False):
        self.run_id, self.fixture = run_id, fixture
        self.manifest = dict(manifest or {})
        self.high_watermark_seq = 0
        self.engine_status = "loading"
        self.trace_complete = True
        self.block_size = self.pool_blocks = 0
        self.owners: dict[int, str] = {}
        self.requests: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}

    def apply(self, e: dict) -> bool:
        """Apply one event; False for a duplicate. Raises SequenceGap if events were skipped."""
        if e["seq"] <= self.high_watermark_seq:
            return False
        if e["seq"] != self.high_watermark_seq + 1:
            raise SequenceGap(f"expected seq {self.high_watermark_seq + 1}, got {e['seq']}")
        self.run_id = self.run_id or e["run_id"]
        getattr(self, "_" + e["type"].replace(".", "_"))(e, e["payload"], e["monotonic_ns"])
        self.high_watermark_seq = e["seq"]
        return True

    def _req(self, e):
        return self.requests[e["request_id"]]

    def _engine_ready(self, e, p, ns):
        self.engine_status = "ready"
        self.block_size, self.pool_blocks = p["block_size"], p["pool_blocks"]
        self.manifest = {**p, **self.manifest}

    def _engine_failed(self, e, p, ns):
        self.engine_status = "failed"
        self.trace_complete = p["trace_complete"]

    def _request_accepted(self, e, p, ns):
        self.requests[e["request_id"]] = _new_request(e["request_id"], ns, p)

    def _request_tokenization_started(self, e, p, ns):
        self._req(e)["status"] = "tokenizing"

    def _request_tokenized(self, e, p, ns):
        self._req(e)["prompt_tokens"] = p["prompt_tokens"]

    def _request_queued(self, e, p, ns):
        r = self._req(e)
        r["status"] = "queued"
        if r["queued_ns"] is None:  # a queue update is not a new queue entry
            r["queued_ns"] = ns
        r["wait_reason"], r["queue_position"] = p["reason"], p["position"]

    def _request_admitted(self, e, p, ns):
        r = self._req(e)
        r["admitted_ns"], r["reservation_blocks"] = ns, p["reservation_blocks"]
        r["wait_reason"] = r["queue_position"] = None

    def _cache_allocated(self, e, p, ns):
        r = self._req(e)
        for m in p["mappings"]:
            assert m["logical_block"] == len(r["block_table"]), "logical table must grow in order"
            r["block_table"].append(m["physical_block"])
            self.owners[m["physical_block"]] = r["request_id"]

    def _cache_released(self, e, p, ns):
        r = self._req(e)
        for b in p["physical_blocks"]:
            self.owners.pop(b, None)
        r["block_table"], r["cached_tokens"], r["reservation_blocks"] = [], 0, 0

    def _batch_scheduled(self, e, p, ns):
        self.batches[e["batch_id"]] = {
            "batch_id": e["batch_id"], "phase": p["phase"], "members": p["members"],
            "request_ids": [m["request_id"] for m in p["members"]], "status": "scheduled",
            "scheduled_ns": ns, "started_ns": None, "completed_ns": None, "start_seq": None, "end_seq": None,
            "host_duration_ms": None, "gpu_duration_ms": None,
        }

    def _batch_started(self, e, p, ns):
        b = self.batches[e["batch_id"]]
        b.update(status="active", started_ns=ns, start_seq=e["seq"])

    def _batch_completed(self, e, p, ns):
        b = self.batches[e["batch_id"]]
        b.update(status="completed", completed_ns=ns, end_seq=e["seq"],
                 host_duration_ms=p["host_duration_ms"], gpu_duration_ms=p["gpu_duration_ms"])

    def _batch_failed(self, e, p, ns):
        self.batches[e["batch_id"]].update(status="failed", completed_ns=ns, end_seq=e["seq"])
        for rid in self.batches[e["batch_id"]]["request_ids"]:
            self.requests[rid]["executing_batch_id"] = None

    def _request_prefill_started(self, e, p, ns):
        r = self._req(e)
        r.update(status="prefill", executing_batch_id=e["batch_id"], prefill_started_ns=ns)

    def _request_prefill_completed(self, e, p, ns):
        r = self._req(e)
        r.update(status="decoding", executing_batch_id=None,
                 processed_tokens=p["processed_tokens"], cached_tokens=p["processed_tokens"])

    def _request_decode_started(self, e, p, ns):
        self._req(e).update(status="decoding", executing_batch_id=e["batch_id"])

    def _request_decode_completed(self, e, p, ns):
        r = self._req(e)
        r.update(executing_batch_id=None, decode_step=p["decode_step"],
                 processed_tokens=p["processed_tokens"], cached_tokens=p["processed_tokens"])

    def _token_sampled(self, e, p, ns):
        r = self._req(e)
        assert p["token_index"] == len(r["token_ids"]), "token indices are dense and ordered"
        r["token_ids"].append(p["token_id"])
        r["output_text"] += p["text_delta"]
        r["sampled_ns"].append(ns)
        r["generated_tokens"] = p["generated_tokens"]  # sampling never changes cached_tokens

    def _request_cancel_requested(self, e, p, ns):
        self._req(e)["cancel_requested"] = True

    def _request_terminal(self, e, p, ns):
        r = self._req(e)
        r.update(status=p["status"], stop_reason=p["stop_reason"], terminal_ns=ns, error_code=p["error_code"],
                 output_text=p["final_output_text"], generated_tokens=p["generated_tokens"],
                 processed_tokens=p["processed_tokens"], wait_reason=None, queue_position=None,
                 executing_batch_id=None)

    def snapshot(self) -> dict:
        live = [r for r in self.requests.values() if r["terminal_ns"] is None]
        return copy.deepcopy({
            "schema_version": 1,
            "run_id": self.run_id,
            "high_watermark_seq": self.high_watermark_seq,
            "trace_complete": self.trace_complete,
            "fixture": self.fixture,
            "manifest": self.manifest,
            "engine_status": self.engine_status,
            "requests": list(self.requests.values()),
            "cache": {
                "block_size": self.block_size,
                "pool_blocks": self.pool_blocks,
                "free_blocks": [b for b in range(self.pool_blocks) if b not in self.owners],
                "owners": {str(b): rid for b, rid in self.owners.items()},
                "reserved_capacity_blocks": sum(r["reservation_blocks"] for r in live),
            },
            "batches": list(self.batches.values()),
            "metrics": self.metrics(),
        })

    def metrics(self) -> dict:
        owned = len(self.owners)
        valid = sum(min(r["cached_tokens"], len(r["block_table"]) * self.block_size) for r in self.requests.values())
        slots = owned * self.block_size
        return {
            "owned_blocks": owned,
            "free_blocks": self.pool_blocks - owned,
            "unused_owned_slots": slots - valid,
            "internal_slot_waste": (slots - valid) / slots if slots else 0.0,
            "requests": {rid: request_metrics(r) for rid, r in self.requests.items()},
        }


def request_metrics(r: dict) -> dict:
    """Server-side timings on the engine's monotonic clock (TRD timing definitions)."""
    ms = lambda a, b: None if a is None or b is None else (b - a) / 1e6  # noqa: E731
    s = r["sampled_ns"]
    return {
        "server_ttft_ms": ms(r["accepted_ns"], s[0] if s else None),
        "end_to_end_ms": ms(r["accepted_ns"], r["terminal_ns"]),
        "queue_ms": ms(r["queued_ns"], r["prefill_started_ns"]),
        "inter_token_ms": [ms(a, b) for a, b in zip(s, s[1:])],
    }


def replay(events, **kw) -> Projection:
    p = Projection(**kw)
    for e in events:
        p.apply(e)
    return p
