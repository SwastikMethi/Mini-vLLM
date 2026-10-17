import { relMs, type BatchState, type RequestState, type RequestStatus, type State } from '../state/reducer'

export const pid = (p: number) => `P${String(p).padStart(2, '0')}`

export const fmtMs = (s: State, ns: number | null | undefined) => (ns == null ? '—' : `${relMs(s, ns).toFixed(1)} ms`)

export const PHASE_LABEL: Record<RequestStatus, string> = {
  accepted: 'Accepted',
  tokenizing: 'Tokenizing',
  queued: 'Waiting',
  prefill: 'Prefill',
  decoding: 'Decode',
  completed: 'Finished',
  cancelled: 'Cancelled',
  failed: 'Failed',
}

export const WAIT_LABEL = { awaiting_scheduler: 'scheduler', capacity: 'capacity', active_limit: 'active limit' }

export const isTerminal = (r: RequestState) => r.terminal_ns !== null

export function counts(s: State) {
  const rs = Object.values(s.requests)
  return {
    active: rs.filter((r) => r.status === 'prefill' || r.status === 'decoding').length,
    waiting: rs.filter((r) => r.status === 'queued' || r.status === 'accepted' || r.status === 'tokenizing').length,
  }
}

/** Input positions a batch writes for one request, as [first, last]. */
export function batchRange(b: BatchState | undefined, request_id: string): [number, number] | null {
  const m = b?.members.find((x) => x.request_id === request_id)
  return m ? [m.start_position, m.start_position + m.input_tokens - 1] : null
}

/** Planned (not yet committed) writes of the request's currently executing batch. */
export function plannedRange(s: State, request_id: string) {
  const r = s.requests[request_id]
  return r?.executing_batch_id ? batchRange(s.batches[r.executing_batch_id], request_id) : null
}

export const within = (pos: number, range: [number, number] | null) => !!range && pos >= range[0] && pos <= range[1]
