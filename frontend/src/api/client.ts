// Live backend client: commands over HTTP, one global resumable SSE stream per run.
// The stream starts from seq 0 so replay has the whole run; EventSource reconnects send Last-Event-ID.
// A sequence gap or a server `reset` falls back to the atomic snapshot, then resumes after its watermark.
import type { EngineEvent } from '../types/events'
import type { Snapshot } from '../state/reducer'
import type { Store } from '../state/store'

export interface Health {
  status: 'loading' | 'ready' | 'failed'
  run_id: string | null
  model_id: string
  device: string
  error: string | null
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message)
  }
}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new ApiError(res.status, data.error?.code ?? 'http_error', data.error?.message ?? res.statusText)
  return data as T
}

export const api = {
  health: () => call<Health>('GET', '/api/health'),
  snapshot: (runId: string) => call<Snapshot>('GET', `/api/runs/${runId}/snapshot`),
  submit: (prompt: string, maxNewTokens: number) =>
    call<{ request_id: string }>('POST', '/api/requests', {
      prompt,
      max_new_tokens: maxNewTokens,
      client_request_id: crypto.randomUUID(),
    }),
  cancel: (requestId: string) => call<{ status: string }>('POST', `/api/requests/${requestId}/cancel`),
}

/** Stream controls for the UI (Reconnect) and the release check (simulated disconnect). */
export const stream = {
  reconnect: () => {},
  disconnect: () => {},
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** Waits for a ready engine, then keeps the store synchronized with its run. */
export async function connect(store: Store): Promise<void> {
  let health: Health
  for (;;) {
    try {
      health = await api.health()
      store.setEngine(health)
      if (health.status !== 'loading' && health.run_id) break
    } catch {
      store.setConnection('stale')
    }
    await sleep(1000)
  }
  const runId = health.run_id!
  let source: EventSource | null = null

  function open(afterSeq: number | null) {
    source?.close()
    store.setConnection('connecting')
    const url = `/api/runs/${runId}/events${afterSeq === null ? '' : `?after_seq=${afterSeq}`}`
    const es = (source = new EventSource(url))
    let dropped = false
    es.onopen = async () => {
      // A self-retried stream may land on a server that restarted onto a new run.
      if (dropped && (await api.health().catch(() => null))?.run_id !== runId) return location.reload()
      store.setConnection('open')
    }
    es.onerror = () => {
      dropped = true
      store.setConnection('stale')
      // EventSource retries network drops itself (with Last-Event-ID) but gives up on an HTTP error,
      // e.g. a proxy 502 while the server restarts. Then we reopen through a health check.
      if (es.readyState === EventSource.CLOSED) setTimeout(reopen, 1000)
    }
    es.addEventListener('engine_event', (msg) => {
      store.markReceived(performance.now())
      store.ingest([JSON.parse((msg as MessageEvent).data) as EngineEvent])
      if (store.current().resync_needed) void resync()
    })
    es.addEventListener('reset', () => void resync())
  }

  /** Resume after the newest applied event, unless the server restarted onto a new run. */
  async function reopen() {
    const h = await api.health().catch(() => null)
    if (!h) return void setTimeout(reopen, 1000) // server still down
    if (h.run_id !== runId) return location.reload()
    open(store.current().high_watermark_seq)
  }

  let resyncing = false
  async function resync() {
    if (resyncing) return
    resyncing = true
    source?.close()
    store.setConnection('stale')
    try {
      if ((await api.health()).run_id !== runId) return location.reload() // server restarted: new run
      const snap = await api.snapshot(runId)
      store.loadSnapshot(snap)
      open(snap.high_watermark_seq)
    } catch {
      await sleep(1000)
      resyncing = false
      return resync()
    }
    resyncing = false
  }

  // Watchdog: a dev proxy can keep the browser's SSE socket open after the server dies, and SSE heartbeats
  // are comments JS never sees, so health is polled to notice a dead server or a restart onto a new run.
  setInterval(async () => {
    const h = await api.health().catch(() => null)
    if (h) store.setEngine(h)
    if (h && h.run_id !== runId) return location.reload()
    if (!h && source && source.readyState !== EventSource.CLOSED) {
      source?.close()
      store.setConnection('stale')
      void reopen()
    }
  }, 5000)

  stream.reconnect = () => void reopen()
  stream.disconnect = () => {
    source?.close()
    store.setConnection('stale')
  }
  open(null)
}
