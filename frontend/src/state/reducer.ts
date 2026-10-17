// One ordered reducer for lifecycle, block ownership and batch history.
// Follows mini_vllm_docs/contracts/ui_projection_contract.md. Pure: every
// function returns a new State and never mutates its input.
import type {
  BatchFailedPayload,
  BatchScheduledPayload,
  CacheReleasedPayload,
  EngineEvent,
  EngineFailedPayload,
  EngineReadyPayload,
  RequestQueuedPayload,
  RequestTerminalPayload,
} from '../types/events'

export type RequestStatus =
  | 'accepted'
  | 'tokenizing'
  | 'queued'
  | 'prefill'
  | 'decoding'
  | 'completed'
  | 'cancelled'
  | 'failed'
export type Phase = BatchScheduledPayload['phase']
export type Member = BatchScheduledPayload['members'][number]

export interface RequestState {
  request_id: string
  status: RequestStatus
  prompt: string | null
  max_new_tokens: number
  prompt_tokens: number | null
  accepted_ns: number
  queued_ns: number | null
  admitted_ns: number | null
  wait_reason: RequestQueuedPayload['reason'] | null
  queue_position: number | null
  executing_batch_id: string | null
  processed_tokens: number
  cached_tokens: number
  generated_tokens: number
  decode_step: number
  /** Decode step that has started but not yet committed. */
  started_decode_step: number | null
  token_ids: number[]
  output_text: string
  block_table: number[]
  reservation_blocks: number
  sampled_ns: number[]
  cancel_requested: boolean
  terminal_ns: number | null
  stop_reason: RequestTerminalPayload['stop_reason'] | null
  error_code: string | null
  release_reason: CacheReleasedPayload['reason'] | null
}

export interface BatchState {
  batch_id: string
  phase: Phase
  members: Member[]
  request_ids: string[]
  status: 'scheduled' | 'active' | 'completed' | 'failed'
  scheduled_ns: number
  started_ns: number | null
  completed_ns: number | null
  start_seq: number | null
  end_seq: number | null
  host_duration_ms: number | null
  gpu_duration_ms: number | null
  error: Pick<BatchFailedPayload, 'error_code' | 'message'> | null
}

export interface Selection {
  request_id: string | null
  batch_id: string | null
  physical_block: number | null
}

export interface State {
  run_id: string | null
  manifest: (EngineReadyPayload & { trace_mode?: string }) | null
  engine_status: 'loading' | 'ready' | 'failed'
  engine_error: EngineFailedPayload | null
  /** Last applied event sequence. */
  high_watermark_seq: number
  /** Timestamp of the run's first event; timeline times are relative to it. */
  origin_ns: number | null
  /** Timestamp of the last applied event: the observation cursor. */
  cursor_ns: number | null
  /** A sequence gap or foreign run was seen; nothing applies until applySnapshot. */
  resync_needed: boolean
  requests: Record<string, RequestState>
  request_order: string[]
  cache: { block_size: number; pool_blocks: number; owners: Record<number, string> }
  batches: Record<string, BatchState>
  batch_order: string[]
  selection: Selection
}

export interface Snapshot {
  schema_version: 1
  run_id: string
  high_watermark_seq: number
  trace_complete: boolean
  fixture: boolean
  manifest: Record<string, unknown>
  engine_status: State['engine_status']
  requests: Partial<RequestState>[]
  cache: {
    block_size: number
    pool_blocks: number
    free_blocks: number[]
    owners: Record<string, string>
    reserved_capacity_blocks: number
  }
  batches: Partial<BatchState>[]
  metrics: Record<string, unknown>
}

export type Action =
  | { kind: 'event'; event: EngineEvent }
  | { kind: 'snapshot'; snapshot: Snapshot }
  | { kind: 'select'; selection: Partial<Selection> }

const NO_SELECTION: Selection = { request_id: null, batch_id: null, physical_block: null }

export function initialState(): State {
  return {
    run_id: null,
    manifest: null,
    engine_status: 'loading',
    engine_error: null,
    high_watermark_seq: 0,
    origin_ns: null,
    cursor_ns: null,
    resync_needed: false,
    requests: {},
    request_order: [],
    cache: { block_size: 16, pool_blocks: 0, owners: {} },
    batches: {},
    batch_order: [],
    selection: NO_SELECTION,
  }
}

function newRequest(request_id: string, accepted_ns: number): RequestState {
  return {
    request_id,
    status: 'accepted',
    prompt: null,
    max_new_tokens: 0,
    prompt_tokens: null,
    accepted_ns,
    queued_ns: null,
    admitted_ns: null,
    wait_reason: null,
    queue_position: null,
    executing_batch_id: null,
    processed_tokens: 0,
    cached_tokens: 0,
    generated_tokens: 0,
    decode_step: 0,
    started_decode_step: null,
    token_ids: [],
    output_text: '',
    block_table: [],
    reservation_blocks: 0,
    sampled_ns: [],
    cancel_requested: false,
    terminal_ns: null,
    stop_reason: null,
    error_code: null,
    release_reason: null,
  }
}

export function reduce(state: State, action: Action): State {
  switch (action.kind) {
    case 'event':
      return applyEvent(state, action.event)
    case 'snapshot':
      return applySnapshot(action.snapshot, state.selection)
    case 'select':
      return { ...state, selection: { ...state.selection, ...action.selection } }
  }
}

/** Applies one event in order. Duplicates are ignored; gaps flag resync_needed. */
export function applyEvent(state: State, e: EngineEvent): State {
  if (state.resync_needed) return state
  if (state.run_id !== null && e.run_id !== state.run_id) return { ...state, resync_needed: true }
  if (e.seq <= state.high_watermark_seq) return state
  if (e.seq !== state.high_watermark_seq + 1) return { ...state, resync_needed: true }
  return project(
    {
      ...state,
      run_id: e.run_id,
      high_watermark_seq: e.seq,
      origin_ns: state.origin_ns ?? e.monotonic_ns,
      cursor_ns: e.monotonic_ns,
    },
    e,
  )
}

type Patch<T> = Partial<T> | ((prev: T) => Partial<T>)

function patchRequest(s: State, id: string | null, patch: Patch<RequestState>): State {
  const prev = id === null ? undefined : s.requests[id]
  if (!prev) return s
  const next = { ...prev, ...(typeof patch === 'function' ? patch(prev) : patch) }
  return { ...s, requests: { ...s.requests, [prev.request_id]: next } }
}

function patchBatch(s: State, id: string | null, patch: Partial<BatchState>): State {
  const prev = id === null ? undefined : s.batches[id]
  if (!prev) return s
  return { ...s, batches: { ...s.batches, [prev.batch_id]: { ...prev, ...patch } } }
}

/** Ends a batch and clears executing_batch_id on its members. */
function endBatch(s: State, id: string, patch: Partial<BatchState>): State {
  const b = s.batches[id]
  if (!b) return s
  let next = patchBatch(s, id, patch)
  for (const rid of b.request_ids) {
    if (next.requests[rid]?.executing_batch_id === id) next = patchRequest(next, rid, { executing_batch_id: null })
  }
  return next
}

function project(s: State, e: EngineEvent): State {
  const t = e.monotonic_ns
  switch (e.type) {
    case 'engine.ready': {
      const { block_size, pool_blocks } = e.payload
      return { ...s, manifest: e.payload, engine_status: 'ready', cache: { block_size, pool_blocks, owners: {} } }
    }
    case 'engine.failed':
      return { ...s, engine_status: 'failed', engine_error: e.payload }

    case 'request.accepted': {
      if (s.requests[e.request_id]) return s
      const r = { ...newRequest(e.request_id, t), prompt: e.payload.prompt, max_new_tokens: e.payload.max_new_tokens }
      return { ...s, requests: { ...s.requests, [r.request_id]: r }, request_order: [...s.request_order, r.request_id] }
    }
    case 'request.tokenization.started':
      return patchRequest(s, e.request_id, { status: 'tokenizing' })
    case 'request.tokenized':
      return patchRequest(s, e.request_id, { prompt_tokens: e.payload.prompt_tokens })
    case 'request.queued':
      // Queue age uses the first queue entry; later updates change reason and position only.
      return patchRequest(s, e.request_id, (r) => ({
        status: 'queued',
        queued_ns: r.queued_ns ?? t,
        wait_reason: e.payload.reason,
        queue_position: e.payload.position,
      }))
    case 'request.admitted':
      return patchRequest(s, e.request_id, {
        status: 'prefill',
        admitted_ns: t,
        reservation_blocks: e.payload.reservation_blocks,
        wait_reason: null,
        queue_position: null,
      })

    case 'cache.allocated': {
      // Appends mappings and changes ownership. Does not commit any slot.
      const owners = { ...s.cache.owners }
      for (const m of e.payload.mappings) owners[m.physical_block] = e.request_id
      const next = { ...s, cache: { ...s.cache, owners } }
      return patchRequest(next, e.request_id, (r) => {
        const block_table = [...r.block_table]
        for (const m of e.payload.mappings) block_table[m.logical_block] = m.physical_block
        return { block_table }
      })
    }
    case 'cache.released': {
      const owners = { ...s.cache.owners }
      for (const p of e.payload.physical_blocks) if (owners[p] === e.request_id) delete owners[p]
      const next = { ...s, cache: { ...s.cache, owners } }
      // processed_tokens stays as history.
      return patchRequest(next, e.request_id, {
        block_table: [],
        cached_tokens: 0,
        reservation_blocks: 0,
        release_reason: e.payload.reason,
      })
    }

    case 'batch.scheduled': {
      if (s.batches[e.batch_id]) return s
      const b: BatchState = {
        batch_id: e.batch_id,
        phase: e.payload.phase,
        members: e.payload.members,
        request_ids: e.payload.members.map((m) => m.request_id),
        status: 'scheduled',
        scheduled_ns: t,
        started_ns: null,
        completed_ns: null,
        start_seq: null,
        end_seq: null,
        host_duration_ms: null,
        gpu_duration_ms: null,
        error: null,
      }
      return { ...s, batches: { ...s.batches, [b.batch_id]: b }, batch_order: [...s.batch_order, b.batch_id] }
    }
    case 'batch.started': {
      const b = s.batches[e.batch_id]
      if (!b) return s
      let next = patchBatch(s, b.batch_id, { status: 'active', started_ns: t, start_seq: e.seq })
      for (const rid of b.request_ids) next = patchRequest(next, rid, { executing_batch_id: b.batch_id })
      return next
    }
    case 'batch.completed':
      return endBatch(s, e.batch_id, {
        status: 'completed',
        completed_ns: t,
        end_seq: e.seq,
        host_duration_ms: e.payload.host_duration_ms,
        gpu_duration_ms: e.payload.gpu_duration_ms ?? null,
      })
    case 'batch.failed':
      return endBatch(s, e.batch_id, {
        status: 'failed',
        completed_ns: t,
        end_seq: e.seq,
        error: { error_code: e.payload.error_code, message: e.payload.message },
      })

    case 'request.prefill.started':
      return patchRequest(s, e.request_id, { status: 'prefill' })
    case 'request.prefill.completed':
      return patchRequest(s, e.request_id, {
        processed_tokens: e.payload.processed_tokens,
        cached_tokens: e.payload.processed_tokens,
      })
    case 'request.decode.started':
      return patchRequest(s, e.request_id, { status: 'decoding', started_decode_step: e.payload.decode_step })
    case 'request.decode.completed':
      return patchRequest(s, e.request_id, {
        decode_step: e.payload.decode_step,
        started_decode_step: null,
        processed_tokens: e.payload.processed_tokens,
        cached_tokens: e.payload.processed_tokens,
      })
    case 'token.sampled':
      // Output only: sampling never fills a KV slot.
      return patchRequest(s, e.request_id, (r) => {
        const sampled_ns = [...r.sampled_ns]
        const token_ids = [...r.token_ids]
        sampled_ns[e.payload.token_index] = t
        token_ids[e.payload.token_index] = e.payload.token_id
        return {
          sampled_ns,
          token_ids,
          output_text: r.output_text + e.payload.text_delta,
          generated_tokens: e.payload.generated_tokens,
        }
      })

    case 'request.cancel_requested':
      return patchRequest(s, e.request_id, { cancel_requested: true })
    case 'request.terminal':
      return patchRequest(s, e.request_id, {
        status: e.payload.status,
        stop_reason: e.payload.stop_reason,
        terminal_ns: t,
        generated_tokens: e.payload.generated_tokens,
        processed_tokens: e.payload.processed_tokens,
        output_text: e.payload.final_output_text,
        error_code: e.payload.error_code,
        wait_reason: null,
        queue_position: null,
        executing_batch_id: null,
        started_decode_step: null,
      })
  }
}

/** Rebuilds the state from an atomic snapshot; subscribe after its high_watermark_seq. */
export function applySnapshot(snap: Snapshot, selection: Selection = NO_SELECTION): State {
  const requests: Record<string, RequestState> = {}
  for (const r of snap.requests) {
    const id = r.request_id as string
    requests[id] = { ...newRequest(id, r.accepted_ns ?? 0), ...r }
  }
  const batches: Record<string, BatchState> = {}
  for (const b of snap.batches) {
    const base = { error: null, start_seq: null, end_seq: null, host_duration_ms: null, gpu_duration_ms: null }
    batches[b.batch_id as string] = { ...base, ...b } as BatchState
  }
  const owners: Record<number, string> = {}
  for (const [k, v] of Object.entries(snap.cache.owners)) owners[Number(k)] = v
  const times = [
    ...snap.requests.flatMap((r) => [r.accepted_ns, r.terminal_ns, ...(r.sampled_ns ?? [])]),
    ...snap.batches.flatMap((b) => [b.scheduled_ns, b.completed_ns]),
  ].filter((n): n is number => typeof n === 'number')
  return {
    run_id: snap.run_id,
    manifest: snap.manifest as unknown as State['manifest'],
    engine_status: snap.engine_status,
    engine_error: null,
    high_watermark_seq: snap.high_watermark_seq,
    origin_ns: times.length ? Math.min(...times) : null,
    cursor_ns: times.length ? Math.max(...times) : null,
    resync_needed: false,
    requests,
    request_order: snap.requests.map((r) => r.request_id as string),
    cache: { block_size: snap.cache.block_size, pool_blocks: snap.cache.pool_blocks, owners },
    batches,
    batch_order: snap.batches.map((b) => b.batch_id as string),
    selection,
  }
}

/** Folds journal events with seq <= `seq` onto `base` (a fresh run or an applied snapshot). */
export function stateAt(events: readonly EngineEvent[], seq: number, base: State = initialState()): State {
  let s = base
  for (const e of events) {
    if (e.seq > seq) break
    s = applyEvent(s, e)
  }
  return s
}

// ---- Derived projections --------------------------------------------------

/** Committed slots in logical block `j` of a request. */
export function validSlots(r: RequestState, j: number, block_size: number): number {
  return Math.min(block_size, Math.max(0, r.cached_tokens - j * block_size))
}

export interface BlockView {
  physical_block: number
  owner: string | null
  logical_block: number | null
  valid_slots: number
}

/** Physical pool in physical-ID order. */
export function physicalBlocks(s: State): BlockView[] {
  const { block_size, pool_blocks, owners } = s.cache
  return Array.from({ length: pool_blocks }, (_, p) => {
    const owner = owners[p] ?? null
    const r = owner ? s.requests[owner] : undefined
    const j = r ? r.block_table.indexOf(p) : -1
    return {
      physical_block: p,
      owner,
      logical_block: j >= 0 ? j : null,
      valid_slots: r && j >= 0 ? validSlots(r, j, block_size) : 0,
    }
  })
}

export function cacheSummary(s: State) {
  const blocks = physicalBlocks(s)
  const owned = blocks.filter((b) => b.owner !== null)
  return {
    owned_blocks: owned.length,
    free_blocks: blocks.filter((b) => b.owner === null).map((b) => b.physical_block),
    unused_owned_slots: owned.reduce((n, b) => n + s.cache.block_size - b.valid_slots, 0),
    reserved_capacity_blocks: Object.values(s.requests).reduce((n, r) => n + r.reservation_blocks, 0),
  }
}

/** Projects the state into the snapshot shape (event-derived fields only). */
export function toSnapshot(s: State) {
  const { free_blocks, reserved_capacity_blocks } = cacheSummary(s)
  return {
    schema_version: 1 as const,
    run_id: s.run_id,
    high_watermark_seq: s.high_watermark_seq,
    engine_status: s.engine_status,
    manifest: s.manifest,
    requests: s.request_order.map((id) => s.requests[id]),
    cache: {
      block_size: s.cache.block_size,
      pool_blocks: s.cache.pool_blocks,
      free_blocks,
      owners: Object.fromEntries(Object.entries(s.cache.owners)),
      reserved_capacity_blocks,
    },
    batches: s.batch_order.map((id) => s.batches[id]),
  }
}

export const relMs = (s: State, ns: number) => (ns - (s.origin_ns ?? 0)) / 1e6
