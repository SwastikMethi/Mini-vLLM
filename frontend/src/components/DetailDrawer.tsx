import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { relMs } from '../state/reducer'
import { useStore } from '../state/store'
import { fmtMs, isTerminal, PHASE_LABEL, pid, WAIT_LABEL } from './format'

export function DetailDrawer({ onClose }: { onClose: () => void }) {
  const [{ state, live, journal, source }] = useStore()
  const [cancelError, setCancelError] = useState<string | null>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  useEffect(() => closeRef.current?.focus(), [])
  const id = state.selection.request_id
  const r = id ? state.requests[id] : undefined

  const memberOf = new Set(
    Object.values(state.batches)
      .filter((b) => id && b.request_ids.includes(id))
      .map((b) => b.batch_id),
  )
  const events = journal.filter(
    (e) => e.seq <= state.high_watermark_seq && (e.request_id === id || (e.batch_id && memberOf.has(e.batch_id))),
  )
  const since = (a: number | null | undefined, b: number | null | undefined) =>
    a == null || b == null ? '—' : `${(relMs(state, b) - relMs(state, a)).toFixed(1)} ms`

  return (
    <aside className="drawer" role="dialog" aria-label={`Request ${id ?? ''} details`}>
      <div className="pane-head">
        <h2 className="mono">{r?.request_id ?? 'No request selected'}</h2>
        {r && <span className={`badge phase-${r.status}`}>{PHASE_LABEL[r.status]}</span>}
        <span className="spacer" />
        <button ref={closeRef} className="btn btn-small" onClick={onClose}>
          Close (Esc)
        </button>
      </div>
      {r && (
        <div className="drawer-body">
          <section>
            <h3 className="label">Output</h3>
            <pre className="output mono">{r.output_text || '(no output yet)'}</pre>
            {r.prompt && (
              <>
                <h3 className="label">Prompt</h3>
                <pre className="mono">{r.prompt}</pre>
              </>
            )}
          </section>

          <section>
            <h3 className="label">State at seq {state.high_watermark_seq}</h3>
            <dl className="dl mono">
              <dt>Stop reason</dt>
              <dd>{r.stop_reason ?? '—'}</dd>
              <dt>Cancellation</dt>
              <dd>{r.cancel_requested ? (isTerminal(r) ? 'requested · cleaned up' : 'requested · pending cleanup') : 'none'}</dd>
              {r.error_code && (
                <>
                  <dt>Error</dt>
                  <dd className="error">{r.error_code}</dd>
                </>
              )}
              <dt>Wait</dt>
              <dd>{r.wait_reason ? `${WAIT_LABEL[r.wait_reason]} · position ${r.queue_position}` : '—'}</dd>
              <dt>Executing</dt>
              <dd>{r.executing_batch_id ?? 'no forward running'}</dd>
              <dt>Prompt tokens</dt>
              <dd>{r.prompt_tokens ?? '—'}</dd>
              <dt>Processed</dt>
              <dd>{r.processed_tokens}</dd>
              <dt>Cached (committed)</dt>
              <dd>{r.cached_tokens}</dd>
              <dt>Generated</dt>
              <dd>
                {r.generated_tokens} / {r.max_new_tokens || '—'}
              </dd>
              <dt>Decode step</dt>
              <dd>
                {r.decode_step} committed{r.started_decode_step !== null && ` · step ${r.started_decode_step} running`}
              </dd>
              <dt>Blocks</dt>
              <dd>{r.block_table.length ? r.block_table.map(pid).join(' ') : r.release_reason ? `released (${r.release_reason})` : '—'}</dd>
              <dt>Credits</dt>
              <dd>{r.reservation_blocks}</dd>
            </dl>
          </section>

          <section>
            <h3 className="label">Timing (relative to run start)</h3>
            <dl className="dl mono">
              <dt>Accepted</dt>
              <dd>{fmtMs(state, r.accepted_ns)}</dd>
              <dt>First queued</dt>
              <dd>{fmtMs(state, r.queued_ns)}</dd>
              <dt>Admitted</dt>
              <dd>{fmtMs(state, r.admitted_ns)}</dd>
              <dt>First token</dt>
              <dd>{fmtMs(state, r.sampled_ns[0])}</dd>
              <dt>Terminal</dt>
              <dd>{fmtMs(state, r.terminal_ns)}</dd>
              <dt>Queue age</dt>
              <dd>{since(r.queued_ns, r.admitted_ns ?? state.cursor_ns)}</dd>
              <dt>Time to first token</dt>
              <dd>{since(r.accepted_ns, r.sampled_ns[0])}</dd>
            </dl>
          </section>

          <section>
            <h3 className="label">
              Raw events ({events.length}
              {journal.length === 0 && ', history before snapshot unavailable'})
            </h3>
            <ol className="events mono">
              {events.map((e) => (
                <li key={e.seq}>
                  <span className="muted">
                    #{e.seq} {fmtMs(state, e.monotonic_ns)}
                  </span>{' '}
                  {e.type}
                  {e.batch_id && <span className="muted"> {e.batch_id}</span>}{' '}
                  <span className="muted">{JSON.stringify(e.payload)}</span>
                </li>
              ))}
            </ol>
          </section>

          {(() => {
            // Cancellation acts on the live request, never on a historical view of it.
            const now = id ? live.requests[id] : undefined
            const done = !now || isTerminal(now)
            const pending = !!now?.cancel_requested && !done
            return (
              <>
                <button
                  className="btn"
                  disabled={source === 'fixture' || done || pending}
                  title={source === 'fixture' ? 'Cancel needs the live backend' : undefined}
                  onClick={() => id && api.cancel(id).catch((e: Error) => setCancelError(e.message))}
                >
                  {pending ? 'Cancellation requested…' : 'Cancel request'}
                </button>
                {source === 'fixture' && <span className="muted"> Cancel needs the live backend.</span>}
                {cancelError && <span className="error-text"> {cancelError}</span>}
              </>
            )
          })()}
        </div>
      )}
    </aside>
  )
}
