import type { RequestState, State } from '../state/reducer'
import { fmtMs, isTerminal, PHASE_LABEL, pid, WAIT_LABEL } from './format'

interface Props {
  r: RequestState
  state: State
  selected: boolean
  /** Batch ID when this request is a member of the selected batch. */
  inBatch: string | null
  onSelect: () => void
  onInspect: () => void
}

type Step = 'done' | 'current' | 'pending' | 'skipped'

function lifecycle(r: RequestState): [string, Step][] {
  const done = isTerminal(r)
  const prefilled = r.processed_tokens > 0
  const at = (current: boolean, complete: boolean): Step => (complete ? 'done' : current ? 'current' : 'pending')
  return [
    ['Accepted', at(r.status === 'accepted' || r.status === 'tokenizing', r.prompt_tokens !== null)],
    ['Queued', at(r.status === 'queued', r.admitted_ns !== null)],
    ['Prefill', at(r.status === 'prefill', prefilled)],
    ['Decode', done && prefilled && r.decode_step === 0 ? 'skipped' : at(r.status === 'decoding', done && r.decode_step > 0)],
    ['Finished', done ? 'done' : 'pending'],
  ]
}

function detail(r: RequestState, s: State): string {
  if (r.cancel_requested && !isTerminal(r)) return 'Cancellation requested · awaiting cleanup'
  switch (r.status) {
    case 'accepted':
    case 'tokenizing':
      return r.prompt_tokens === null ? 'Tokenizing prompt…' : `Tokenized · ${r.prompt_tokens} prompt tokens`
    case 'queued':
      return `Waiting for ${WAIT_LABEL[r.wait_reason ?? 'awaiting_scheduler']} · position ${r.queue_position ?? '—'} · since ${fmtMs(s, r.queued_ns)}`
    case 'prefill':
      return r.executing_batch_id ? `Prefill forward ${r.executing_batch_id} · ${r.prompt_tokens} inputs` : 'Admitted · awaiting prefill forward'
    case 'decoding': {
      const exec = r.executing_batch_id
        ? `executing ${r.executing_batch_id} (step ${r.started_decode_step})`
        : r.started_decode_step !== null
          ? `step ${r.started_decode_step} awaiting commit`
          : 'between engine steps'
      return `Decode step ${r.decode_step} committed · ${exec}`
    }
    default:
      return `${PHASE_LABEL[r.status]} · ${r.stop_reason ?? 'no stop reason'} · ${r.release_reason ? 'pages released' : 'cleanup pending'}`
  }
}

export function RequestRow({ r, state, selected, inBatch, onSelect, onInspect }: Props) {
  const finished = isTerminal(r)
  return (
    <li
      id={`row-${r.request_id}`}
      role="option"
      aria-selected={selected}
      tabIndex={selected ? 0 : -1}
      className={`row ${selected ? 'is-selected' : ''} ${finished ? 'is-finished' : ''}`}
      onClick={onSelect}
      onDoubleClick={onInspect}
      onKeyDown={(e) => e.key === 'Enter' && onInspect()}
    >
      <div className="row-main">
        <span className="mono row-id" title={r.request_id}>{r.request_id}</span>
        <span className={`badge phase-${r.status}`}>{PHASE_LABEL[r.status]}</span>
        <span className="row-prompt" title={r.prompt ?? undefined}>
          {r.prompt ?? <span className="muted">prompt not in snapshot</span>}
        </span>
        <span className="mono row-cached">
          {r.cached_tokens > 0 ? `${r.cached_tokens} cached` : '—'}
        </span>
        <span className="chips">
          {r.block_table.length === 0 ? (
            <span className="muted mono">—</span>
          ) : (
            r.block_table.map((p) => (
              <span key={p} className="chip mono">
                {pid(p)}
              </span>
            ))
          )}
        </span>
      </div>
      <div className="row-sub mono">
        {r.output_text ? (
          <span className="row-output">› {r.output_text}</span>
        ) : (
          <span className="muted">› no output yet</span>
        )}
        <span className="muted row-detail">{detail(r, state)}</span>
        {inBatch && <span className="badge badge-outline">in {inBatch}</span>}
      </div>
      {selected && (
        <div className="lifecycle">
          {lifecycle(r).map(([label, step]) => (
            <span key={label} className={`step step-${step}`}>
              {label} {step === 'done' ? '✓' : step === 'current' ? '●' : step === 'skipped' ? '(skipped)' : ''}
            </span>
          ))}
          <span className="spacer" />
          <button
            className="btn btn-small"
            onClick={(e) => {
              e.stopPropagation()
              onInspect()
            }}
          >
            Inspect
          </button>
        </div>
      )}
    </li>
  )
}
