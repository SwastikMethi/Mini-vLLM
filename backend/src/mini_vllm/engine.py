"""The engine thread: sole owner of model execution, the KV pool and request state.

One iteration (TRD "One engine iteration"): apply commands (cancellation first) → admit at most one FIFO request
and run its complete prefill → one batched decode over all active requests → commit counters only after the
forward succeeds → stream tokens in stable order → stop checks → release. Idle waits on the command queue.
"""

import queue
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future
from datetime import datetime

import torch

from .config import MODEL_REVISION, Config
from .events import EventLog, Sink
from .kv_cache import KVPool
from .model import encode_chat, manifest as model_manifest
from .request import Request
from .runner import DecodeRow, Runner, sample


class EngineError(Exception):
    def __init__(self, http_status: int, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.http_status, self.code, self.message, self.retryable = http_status, code, message, retryable


class Engine:
    def __init__(self, cfg: Config, model, tok, sink: Sink, run_id: str | None = None):
        self.cfg, self.model, self.tok = cfg, model, tok
        mc = model.config
        self.pool = KVPool(layers=mc.num_hidden_layers, blocks=cfg.pool_blocks, block_size=cfg.block_size,
                           kv_heads=mc.num_key_value_heads, head_dim=mc.head_dim, dtype=cfg.dtype, device=cfg.device)
        self.runner = Runner(model, self.pool)
        self.run_id = run_id or f"run-{datetime.now():%Y%m%d-%H%M%S}"
        self.manifest = {**model_manifest(model, cfg.device), "block_size": cfg.block_size,
                         "pool_blocks": cfg.pool_blocks, "pool_bytes": self.pool.nbytes, "trace_mode": "live"}
        self.log = EventLog(self.run_id, sink, self.manifest)
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.status = "loading"
        self.admitting = True
        self.commands: queue.Queue = queue.Queue()
        self.requests: dict[str, Request] = {}
        self.by_client: dict[str, tuple[dict, str]] = {}
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []  # admission order = stable streaming order
        self.n_batches = 0
        self.thread = threading.Thread(target=self._run, name="engine", daemon=True)

    # ---- API-facing (any thread) ------------------------------------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.commands.put(("stop", None, None))
        self.thread.join()

    def submit(self, body: dict) -> Future:
        return self._command("submit", body)

    def cancel(self, rid: str) -> Future:
        return self._command("cancel", rid)

    def _command(self, kind, arg) -> Future:
        fut: Future = Future()
        if self.status == "failed":
            fut.set_exception(EngineError(503, "engine_failed", "engine has failed; restart for a new run"))
        else:
            self.commands.put((kind, arg, fut))
        return fut

    # ---- engine thread ----------------------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._warmup()
            self.log.emit("engine.ready", **{k: self.manifest[k] for k in
                                             ("model_id", "model_revision", "device", "dtype", "block_size", "pool_blocks")})
            self.log.publish()
            self.status = "ready"
            while self._drain(block=not (self.active or self.waiting)):
                self._iteration()
        except Exception as exc:  # crash outside a forward (e.g. journal failure): fail the run visibly
            self._crash(exc)

    def _warmup(self) -> None:
        self.pool.reserve("__warmup__", 1)
        self.pool.ensure("__warmup__", 2)
        self.runner.prefill("__warmup__", [0, 1]).cpu()
        self.pool.release("__warmup__")

    def _drain(self, block: bool) -> bool:
        cmds = []
        try:
            cmds.append(self.commands.get(block=block))
            while True:
                cmds.append(self.commands.get_nowait())
        except queue.Empty:
            pass
        # Cancellation before admission of further work.
        cmds.sort(key=lambda c: c[0] != "cancel")
        for kind, arg, fut in cmds:
            if kind == "stop":
                return False
            try:
                fut.set_result(self._submit(arg) if kind == "submit" else self._cancel(arg))
            except EngineError as err:
                fut.set_exception(err)
        self.log.publish()
        return True

    def _submit(self, body: dict) -> str:
        if self.status != "ready" or not self.admitting:
            raise EngineError(503, "engine_unavailable", f"engine is {self.status}", retryable=True)
        cid = body["client_request_id"]
        if cid in self.by_client:
            prev, rid = self.by_client[cid]
            if prev == body:
                return rid
            raise EngineError(409, "idempotency_conflict", "client_request_id reused with a different body")
        if len(self.waiting) >= self.cfg.queue_cap:
            raise EngineError(429, "queue_full", f"waiting queue is full ({self.cfg.queue_cap})", retryable=True)
        t_accept = self.log.now()
        ids = encode_chat(self.tok, body["prompt"])
        t_tok = self.log.now()
        budget = len(ids) + body["max_new_tokens"]
        if budget > self.cfg.max_request_tokens:
            raise EngineError(422, "request_too_long", f"{budget} tokens exceeds {self.cfg.max_request_tokens}")
        if self.pool.blocks_for(budget) > self.pool.num_blocks:
            raise EngineError(422, "exceeds_pool", f"needs {self.pool.blocks_for(budget)} blocks; pool has {self.pool.num_blocks}")
        rid = f"RQ-{len(self.requests) + 1:03d}"
        req = Request(rid, cid, body["prompt"], ids, body["max_new_tokens"], body["temperature"], body["seed"],
                      accepted_ns=t_accept, deadline_ns=t_accept + self.cfg.deadline_ms * 1_000_000)
        self.requests[rid], self.by_client[cid] = req, (body, rid)
        # Tokenization is synchronous and sub-millisecond; events carry the observed start/end timestamps.
        self.log.emit("request.accepted", request_id=rid, ns=t_accept, prompt=req.prompt,
                      max_new_tokens=req.max_new_tokens, temperature=req.temperature, seed=req.seed,
                      client_request_id=cid, deadline_ms=self.cfg.deadline_ms)
        self.log.emit("request.tokenization.started", request_id=rid, ns=t_accept)
        self.log.emit("request.tokenized", request_id=rid, ns=t_tok, prompt_tokens=len(ids),
                      tokenizer_revision=MODEL_REVISION, template_id="qwen3-thinking-disabled")
        self.waiting.append(req)
        self._refresh_queue()
        return rid

    def _cancel(self, rid: str) -> str:
        req = self.requests.get(rid)
        if req is None:
            raise EngineError(404, "unknown_request", f"no request {rid} in this run")
        if req.terminal:
            return req.status
        if req.cancel is None:
            req.cancel = "user"
            self.log.emit("request.cancel_requested", request_id=rid, reason="user")
        return "cancel_requested"

    def _iteration(self) -> None:
        now = self.log.now()
        for r in [*self.active, *self.waiting]:
            if r.cancel is None and now > r.deadline_ns:
                r.cancel = "deadline"
                self.log.emit("request.cancel_requested", request_id=r.rid, reason="deadline")
        for r in [*self.active, *self.waiting]:
            if r.cancel:  # we are at a safe step boundary: nothing of r is executing
                self._terminate(r, "cancelled", r.cancel)
        self._refresh_queue()
        self.log.publish()
        if self.waiting and self._blocker(self.waiting[0]) == "awaiting_scheduler":
            self._prefill(self.waiting.popleft())
        if self.active and self.admitting:
            self._decode()

    def _blocker(self, head: Request) -> str:
        if len(self.active) >= self.cfg.max_active:
            return "active_limit"
        if self.pool.credits_free < self.pool.blocks_for(len(head.prompt_ids) + head.max_new_tokens):
            return "capacity"
        return "awaiting_scheduler"

    def _refresh_queue(self) -> None:
        """Emit request.queued only when a waiting request's (reason, position) changes. FIFO: everyone behind the
        head waits on the head's blocker."""
        if not self.waiting:
            return
        reason = self._blocker(self.waiting[0])
        for i, r in enumerate(self.waiting):
            if r.wait != (reason, i):
                r.wait = (reason, i)
                self.log.emit("request.queued", request_id=r.rid, reason=reason, position=i)

    def _batch_id(self) -> str:
        self.n_batches += 1
        return f"b{self.n_batches:02d}"

    def _prefill(self, r: Request) -> None:
        p = len(r.prompt_ids)
        r.reservation = self.pool.blocks_for(p + r.max_new_tokens)
        self.pool.reserve(r.rid, r.reservation)
        self.log.emit("request.admitted", request_id=r.rid, reservation_blocks=r.reservation)
        self._allocate(r, p, "prefill")
        bid = self._batch_id()
        self.log.emit("batch.scheduled", batch_id=bid, phase="prefill",
                      members=[{"request_id": r.rid, "input_tokens": p, "start_position": 0}])
        self.log.emit("request.prefill.started", request_id=r.rid, batch_id=bid, input_tokens=p, cached_prefix_tokens=0)
        self.log.emit("batch.started", batch_id=bid, phase="prefill")
        r.status = "prefill"
        self.active.append(r)
        self.log.publish()  # start events are visible before the forward runs
        t = time.perf_counter()
        try:
            logits = self.runner.prefill(r.rid, r.prompt_ids).cpu()
        except Exception as exc:
            return self._abort(bid, "prefill", [r], exc)
        self.log.emit("batch.completed", batch_id=bid, phase="prefill",
                      host_duration_ms=round((time.perf_counter() - t) * 1e3, 3), gpu_duration_ms=None)
        r.cached, r.status = p, "decoding"
        self.log.emit("request.prefill.completed", request_id=r.rid, batch_id=bid, processed_tokens=p)
        self._sample(r, logits, bid)
        self._stop_check(r)
        self.log.publish()

    def _decode(self) -> None:
        rows = list(self.active)
        for r in rows:
            self._allocate(r, r.cached + 1, "decode_growth")
        bid = self._batch_id()
        self.log.emit("batch.scheduled", batch_id=bid, phase="decode",
                      members=[{"request_id": r.rid, "input_tokens": 1, "start_position": r.cached} for r in rows])
        for r in rows:
            self.log.emit("request.decode.started", request_id=r.rid, batch_id=bid,
                          decode_step=r.decode_step + 1, input_position=r.cached)
        self.log.emit("batch.started", batch_id=bid, phase="decode")
        self.log.publish()
        t = time.perf_counter()
        try:
            logits = self.runner.decode([DecodeRow(r.rid, r.tokens[-1], r.cached) for r in rows]).cpu()
        except Exception as exc:
            return self._abort(bid, "decode", rows, exc)
        self.log.emit("batch.completed", batch_id=bid, phase="decode",
                      host_duration_ms=round((time.perf_counter() - t) * 1e3, 3), gpu_duration_ms=None)
        for r in rows:  # commit: counters advance once per whole-model forward
            r.decode_step += 1
            r.cached += 1
            self.log.emit("request.decode.completed", request_id=r.rid, batch_id=bid,
                          decode_step=r.decode_step, processed_tokens=r.cached)
        for i, r in enumerate(rows):
            self._sample(r, logits[i], bid)
        for r in rows:
            self._stop_check(r)
        self.log.publish()

    def _allocate(self, r: Request, tokens: int, reason: str) -> None:
        start = len(self.pool.tables[r.rid])
        new = self.pool.ensure(r.rid, tokens)
        if new:
            self.log.emit("cache.allocated", request_id=r.rid, reason=reason,
                          mappings=[{"logical_block": start + i, "physical_block": b} for i, b in enumerate(new)])

    def _sample(self, r: Request, logits: torch.Tensor, bid: str) -> None:
        tok = sample(logits, r.temperature, r.generator)
        r.tokens.append(tok)
        text = self.tok.decode(r.tokens, skip_special_tokens=True)
        delta = ""
        if not text.endswith("�"):  # hold back an incomplete UTF-8 character
            delta, r.text = text[len(r.text):], text
        self.log.emit("token.sampled", request_id=r.rid, batch_id=bid, token_index=len(r.tokens) - 1, token_id=tok,
                      text_delta=delta, generated_tokens=len(r.tokens), processed_tokens=r.cached)

    def _stop_check(self, r: Request) -> None:
        if r.tokens[-1] in self.eos:
            self._terminate(r, "completed", "eos")
        elif len(r.tokens) >= r.max_new_tokens:
            self._terminate(r, "completed", "length")

    def _terminate(self, r: Request, status: str, stop_reason: str, error_code: str | None = None) -> None:
        """Release ownership and credits exactly once, then emit the terminal event."""
        if r.rid in self.pool.credits:
            blocks = self.pool.release(r.rid)
            self.log.emit("cache.released", request_id=r.rid, physical_blocks=blocks,
                          reservation_blocks=r.reservation, reason=status)
        r.cached, r.status = 0, status
        if r in self.active:
            self.active.remove(r)
        if r in self.waiting:
            self.waiting.remove(r)
        self.log.emit("request.terminal", request_id=r.rid, status=status, stop_reason=stop_reason,
                      generated_tokens=len(r.tokens), processed_tokens=self._processed(r),
                      final_output_text=self.tok.decode(r.tokens, skip_special_tokens=True), error_code=error_code)

    def _processed(self, r: Request) -> int:
        return len(r.prompt_ids) + r.decode_step if r.tokens else 0

    def _abort(self, bid: str, phase: str, rows: list[Request], exc: Exception) -> None:
        """A failed forward: fail its members, stop admissions, fail the rest, mark the engine failed."""
        code = "device_oom" if "out of memory" in str(exc).lower() else "execution_error"
        traceback.print_exception(exc)
        self.log.emit("batch.failed", batch_id=bid, phase=phase, error_code=code, message=str(exc)[:500])
        self.admitting = False
        for r in [*rows, *[x for x in self.active if x not in rows], *self.waiting]:
            self._terminate(r, "failed", "engine_error", code)
        self.status = "failed"
        self.log.emit("engine.failed", error_code=code, message=str(exc)[:500], trace_complete=True)
        self.log.publish()

    def _crash(self, exc: Exception) -> None:
        traceback.print_exception(exc)
        self.status, self.admitting = "failed", False
        for r in [*self.active, *self.waiting]:
            self.pool.release(r.rid)
        self.log.pending.clear()
        self.log.seq = self.log.projection.high_watermark_seq  # uncommitted events were never streamed
        try:
            self.log.emit("engine.failed", error_code="storage_error" if isinstance(exc, OSError) else "engine_crash",
                          message=str(exc)[:500], trace_complete=False)
            self.log.publish()
        except Exception:
            pass
        while True:  # keep failing commands so API callers never hang
            kind, _, fut = self.commands.get()
            if kind == "stop":
                return
            fut.set_exception(EngineError(503, "engine_failed", "engine has failed; restart for a new run"))
