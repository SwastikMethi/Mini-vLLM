"""SQLite WAL event journal. The engine thread is the only writer; API threads read with their own connections.

Snapshots of retained runs are rebuilt by replaying their events through the same reducer the engine uses.
ponytail: full replay per historical snapshot request; store periodic snapshots if runs grow past ~10^5 events.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .snapshot import Projection, replay

MAX_RUNS, MAX_BYTES = 20, 200 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, status TEXT NOT NULL, started_at TEXT NOT NULL,
  trace_complete INTEGER NOT NULL DEFAULT 1, manifest TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY (run_id, seq)
);
"""


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w = self._connect()
        self.w.executescript(SCHEMA)
        # A run still marked active belongs to a process that died: it can be replayed, never resumed.
        self.w.execute("UPDATE runs SET status='failed' WHERE status='active'")
        self.w.commit()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    # ---- writer (engine thread, plus startup/shutdown) -------------------------------------------------------
    def start_run(self, run_id: str, manifest: dict) -> None:
        self.evict()
        self.w.execute("INSERT INTO runs (run_id, status, started_at, manifest) VALUES (?, 'active', ?, ?)",
                       (run_id, datetime.now(timezone.utc).isoformat(), json.dumps(manifest)))
        self.w.commit()

    def append(self, group: list[dict]) -> None:
        """One transaction per committed event group."""
        rows = [(e["run_id"], e["seq"], json.dumps(e, separators=(",", ":"))) for e in group]
        with self.w:
            self.w.executemany("INSERT INTO events (run_id, seq, body) VALUES (?, ?, ?)", rows)
            self.w.execute("UPDATE runs SET bytes = bytes + ? WHERE run_id = ?",
                           (sum(len(r[2]) for r in rows), group[0]["run_id"]))
            for e in group:
                if e["type"] == "engine.failed":
                    self.w.execute("UPDATE runs SET status='failed', trace_complete=? WHERE run_id=?",
                                   (int(e["payload"]["trace_complete"]), e["run_id"]))

    def finish_run(self, run_id: str) -> None:
        self.w.execute("UPDATE runs SET status='completed' WHERE run_id=? AND status='active'", (run_id,))
        self.w.commit()

    def evict(self) -> None:
        """Keep at most MAX_RUNS / MAX_BYTES of finished runs; oldest finished runs go first."""
        done = self.w.execute(
            "SELECT run_id, bytes FROM runs WHERE status != 'active' ORDER BY started_at DESC").fetchall()
        total, keep = 0, 0
        for i, (_, size) in enumerate(done):
            if i >= MAX_RUNS - 1 or total + size > MAX_BYTES:  # leave room for the run about to start
                break
            total, keep = total + size, i + 1
        with self.w:
            for run_id, _ in done[keep:]:
                self.w.execute("DELETE FROM events WHERE run_id=?", (run_id,))
                self.w.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    # ---- readers (any thread) --------------------------------------------------------------------------------
    def _read(self, q: str, args=()) -> list[tuple]:
        with closing(self._connect()) as c:
            return c.execute(q, args).fetchall()

    def runs(self) -> list[dict]:
        rows = self._read("SELECT run_id, status, started_at, trace_complete FROM runs ORDER BY started_at DESC")
        return [{"run_id": r, "status": s, "started_at": t, "trace_complete": bool(tc)} for r, s, t, tc in rows]

    def run(self, run_id: str) -> dict | None:
        rows = self._read("SELECT status, manifest, trace_complete FROM runs WHERE run_id=?", (run_id,))
        return {"status": rows[0][0], "manifest": json.loads(rows[0][1]), "trace_complete": bool(rows[0][2])} if rows else None

    def events(self, run_id: str, after_seq: int = 0, limit: int = -1) -> list[dict]:
        rows = self._read("SELECT body FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after_seq, limit))
        return [json.loads(b) for (b,) in rows]

    def max_seq(self, run_id: str) -> int:
        return self._read("SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id=?", (run_id,))[0][0]

    def projection(self, run_id: str) -> Projection | None:
        meta = self.run(run_id)
        if meta is None:
            return None
        p = replay(self.events(run_id), run_id=run_id, manifest=meta["manifest"])
        p.trace_complete = p.trace_complete and meta["trace_complete"]
        return p
