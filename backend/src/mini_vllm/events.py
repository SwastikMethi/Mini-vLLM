"""Event envelopes. Events are published to the sink (journal + stream) in groups and folded into the projection only
after the sink accepts them, so /snapshot never shows an uncommitted event."""

import threading
import time
from datetime import datetime, timezone
from typing import Callable

from .snapshot import Projection

Sink = Callable[[list[dict]], None]


class EventLog:
    def __init__(self, run_id: str, sink: Sink, manifest: dict):
        self.run_id, self.sink = run_id, sink
        self.t0 = time.monotonic_ns()
        self.seq = 0
        self.pending: list[dict] = []
        self.projection = Projection(run_id, manifest)
        self.lock = threading.Lock()  # guards projection reads from API threads

    def now(self) -> int:
        return time.monotonic_ns() - self.t0

    def emit(self, type_: str, *, request_id=None, batch_id=None, ns: int | None = None, **payload) -> dict:
        ns = self.now() if ns is None else ns
        wall = datetime.fromtimestamp(time.time() - (self.now() - ns) / 1e9, timezone.utc)
        self.seq += 1
        e = {
            "schema_version": 1, "run_id": self.run_id, "seq": self.seq, "type": type_,
            "monotonic_ns": ns, "wall_time": wall.isoformat().replace("+00:00", "Z"), "source": "engine",
            "request_id": request_id, "batch_id": batch_id, "payload": payload,
        }
        self.pending.append(e)
        return e

    def publish(self) -> None:
        """Commit the pending group. Called before a forward (start events) and after it (results)."""
        if self.pending:
            group, self.pending = self.pending, []
            self.sink(group)
            with self.lock:
                for e in group:
                    self.projection.apply(e)

    def snapshot(self) -> dict:
        with self.lock:
            return self.projection.snapshot()
