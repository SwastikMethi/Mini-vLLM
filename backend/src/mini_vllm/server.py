"""FastAPI app. Handlers enqueue commands for the engine thread and await acknowledgement; no model work runs on the
event loop. Events reach browsers through a journal-backed, resumable global SSE stream.

Run with exactly one worker:  uv run uvicorn mini_vllm.server:app --workers 1
"""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import MODEL_ID, MODEL_REVISION, Config
from .engine import Engine, EngineError
from .journal import Journal
from .model import load
from .snapshot import Projection

HEARTBEAT_S = 15
SUBSCRIBER_QUEUE = 1024  # event groups; a slower browser is disconnected and resumes from the journal
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(128, ge=1, le=256)
    temperature: float = Field(0.0, ge=0, le=2)
    seed: int = Field(0, ge=0)
    client_request_id: str = Field(min_length=1)


def error(status: int, code: str, message: str, retryable: bool = False) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "retryable": retryable}}, status_code=status)


class Subscriber:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(SUBSCRIBER_QUEUE)
        self.overflow = False


class Broker:
    """Fans committed event groups out to SSE subscribers without ever blocking the engine thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop, self.subs = loop, set()

    def subscribe(self) -> Subscriber:
        s = Subscriber()
        self.subs.add(s)
        return s

    def unsubscribe(self, s: Subscriber | None) -> None:
        self.subs.discard(s)

    def publish(self, group: list[dict]) -> None:  # engine thread
        self.loop.call_soon_threadsafe(self._fanout, group)

    def _fanout(self, group: list[dict]) -> None:
        for s in list(self.subs):
            try:
                s.queue.put_nowait(group)
            except asyncio.QueueFull:
                s.overflow = True
                self.subs.discard(s)


class State:
    def __init__(self, cfg: Config, loader):
        self.cfg, self.loader = cfg, loader
        self.journal = Journal(cfg.data_dir / "journal.sqlite")
        self.engine: Engine | None = None
        self.load_error: str | None = None
        self.broker: Broker | None = None

    @property
    def status(self) -> str:
        if self.load_error:
            return "failed"
        return self.engine.status if self.engine else "loading"

    @property
    def run_id(self) -> str | None:
        return self.engine.run_id if self.engine else None

    def start(self, loop) -> None:
        self.broker = Broker(loop)
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            model, tok = self.loader(self.cfg.device)
            eng = Engine(self.cfg, model, tok, self._sink)
            self.journal.start_run(eng.run_id, eng.manifest)
            self.engine = eng
            eng.start()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def _sink(self, group: list[dict]) -> None:
        """Journal first, then stream: a browser never sees an event the journal does not have. The one exception
        is the engine.failed notice after a journal failure, which says trace_complete=false."""
        try:
            self.journal.append(group)
        except Exception:
            if not all(e["type"] == "engine.failed" for e in group):
                raise
        self.broker.publish(group)

    def stop(self) -> None:
        if self.engine:
            self.engine.stop()
            self.journal.finish_run(self.engine.run_id)


def sse(e: dict) -> str:
    return f"id: {e['run_id']}:{e['seq']}\nevent: engine_event\ndata: {json.dumps(e, separators=(',', ':'))}\n\n"


def reset(run_id: str, reason: str) -> str:
    data = json.dumps({"reason": reason, "snapshot_url": f"/api/runs/{run_id}/snapshot"})
    return f"event: reset\ndata: {data}\n\n"


def resolve_cursor(st: State, run_id: str, last_event_id: str | None, after_seq: int | None) -> int | None:
    """Last-Event-ID (run_id:seq) wins over after_seq; no cursor starts from zero. None means reset."""
    if last_event_id:
        rid, _, seq = last_event_id.rpartition(":")
        if rid != run_id or not seq.isdigit():
            return None
        cursor = int(seq)
    else:
        cursor = after_seq or 0
    return cursor if cursor <= st.journal.max_seq(run_id) else None


async def event_stream(st: State, run_id: str, cursor: int, is_disconnected):
    live = run_id == st.run_id
    sub = st.broker.subscribe() if live else None  # subscribe before the backlog read so nothing falls between
    last = cursor
    try:
        for e in await asyncio.to_thread(st.journal.events, run_id, last):
            yield sse(e)
            last = e["seq"]
        while not await is_disconnected():
            if sub is None or sub.overflow:
                if sub is not None:
                    return  # slow subscriber: close; EventSource reconnects with Last-Event-ID from the journal
                await asyncio.sleep(HEARTBEAT_S)
                yield ": heartbeat\n\n"
                continue
            try:
                group = await asyncio.wait_for(sub.queue.get(), HEARTBEAT_S)
            except TimeoutError:
                yield ": heartbeat\n\n"
                continue
            for e in group:
                if e["seq"] <= last:
                    continue  # already sent from the backlog
                if e["seq"] > last + 1:  # backfill from the journal rather than skip
                    for m in await asyncio.to_thread(st.journal.events, run_id, last, e["seq"] - last - 1):
                        yield sse(m)
                yield sse(e)
                last = e["seq"]
    finally:
        st.broker.unsubscribe(sub)


def create_app(cfg: Config | None = None, loader=load) -> FastAPI:
    cfg = cfg or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.st = State(cfg, loader)
        app.state.st.start(asyncio.get_running_loop())
        yield
        await asyncio.to_thread(app.state.st.stop)

    app = FastAPI(title="Mini-vLLM local API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid(_req, exc: RequestValidationError):
        return error(422, "invalid_request", "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))

    @app.exception_handler(EngineError)
    async def engine_error(_req, exc: EngineError):
        return error(exc.http_status, exc.code, exc.message, exc.retryable)

    def st(request: Request) -> State:
        return request.app.state.st

    @app.get("/api/health")
    async def health(request: Request):
        s = st(request)
        manifest = s.engine.manifest if s.engine else {}
        return {"status": s.status, "run_id": s.run_id, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "device": cfg.device, "manifest": manifest, "error": s.load_error}

    @app.get("/api/runs")
    async def runs(request: Request):
        return await asyncio.to_thread(st(request).journal.runs)

    @app.post("/api/requests", status_code=202)
    async def submit(body: Submission, request: Request):
        s = st(request)
        if s.status != "ready":
            return error(503, "engine_unavailable", f"engine is {s.status}", retryable=s.status == "loading")
        rid = await asyncio.wrap_future(s.engine.submit(body.model_dump()))
        return {"request_id": rid, "run_id": s.run_id, "status": "accepted", "events_url": f"/api/runs/{s.run_id}/events"}

    @app.post("/api/requests/{request_id}/cancel")
    async def cancel(request_id: str, request: Request):
        s = st(request)
        if s.engine is None:
            return error(503, "engine_unavailable", f"engine is {s.status}", retryable=True)
        status = await asyncio.wrap_future(s.engine.cancel(request_id))
        return JSONResponse({"request_id": request_id, "status": status}, status_code=202 if status == "cancel_requested" else 200)

    async def projection_snapshot(s: State, run_id: str) -> dict | None:
        if run_id == s.run_id:
            snap = s.engine.log.snapshot()
            return snap
        p = await asyncio.to_thread(s.journal.projection, run_id)
        return p.snapshot() if p else None

    @app.get("/api/runs/{run_id}/snapshot")
    async def snapshot(run_id: str, request: Request):
        snap = await projection_snapshot(st(request), run_id)
        return snap if snap else error(404, "unknown_run", f"run {run_id} is unknown or evicted")

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after_seq: int | None = Query(None, ge=0),
                     last_event_id: str | None = Header(None)):
        s = st(request)
        if await asyncio.to_thread(s.journal.run, run_id) is None:
            return error(404, "unknown_run", f"run {run_id} is unknown or evicted")
        cursor = await asyncio.to_thread(resolve_cursor, s, run_id, last_event_id, after_seq)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        if cursor is None:
            async def stale():
                yield reset(run_id, "stale_or_invalid_cursor")
            return StreamingResponse(stale(), media_type="text/event-stream", headers=headers)
        return StreamingResponse(event_stream(s, run_id, cursor, request.is_disconnected),
                                 media_type="text/event-stream", headers=headers)

    @app.get("/api/runs/{run_id}/export")
    async def export(run_id: str, request: Request):
        s = st(request)
        meta = await asyncio.to_thread(s.journal.run, run_id)
        if meta is None:
            return error(404, "unknown_run", f"run {run_id} is unknown or evicted")
        return {"manifest": meta["manifest"], "initial_state": Projection(run_id, meta["manifest"]).snapshot(),
                "events": await asyncio.to_thread(s.journal.events, run_id),
                "final_snapshot": await projection_snapshot(s, run_id)}

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


app = create_app()
