import { useEffect, useRef, useState } from 'react'
import { relMs, type BatchState, type RequestState, type State } from '../state/reducer'
import { useStore } from '../state/store'
import { fmtMs, PHASE_LABEL, WAIT_LABEL } from './format'

const LANE = 30
const AXIS = 26
const PAD = 12
const ZOOMS = [1, 2, 4, 8, 16, 32]
const TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

function describe(s: State, b: BatchState) {
  const members = b.members.map((m) => `${m.request_id} [${m.start_position}–${m.start_position + m.input_tokens - 1}]`)
  return [
    `${b.batch_id} · ${b.phase} · ${b.status}`,
    `members: ${members.join(', ')}`,
    `input tokens: ${b.members.reduce((n, m) => n + m.input_tokens, 0)}`,
    `start ${fmtMs(s, b.started_ns)} · end ${b.completed_ns === null ? 'ongoing' : fmtMs(s, b.completed_ns)}`,
    `host ${b.host_duration_ms ?? '—'} ms · GPU ${b.gpu_duration_ms ?? 'unavailable'}`,
    b.error ? `error ${b.error.error_code}: ${b.error.message}` : '',
  ]
    .filter(Boolean)
    .join('\n')
}

/** Engine batches that ran while `r` was in its decode lifecycle without it. */
function idleIntervals(s: State, r: RequestState, started: BatchState[], candidates = started) {
  const ownPrefill = started.find((b) => b.phase === 'prefill' && b.request_ids.includes(r.request_id))
  if (!ownPrefill?.completed_ns) return []
  const end = r.terminal_ns ?? s.cursor_ns ?? 0
  return candidates.filter(
    (b) => !b.request_ids.includes(r.request_id) && b.started_ns! >= ownPrefill.completed_ns! && b.started_ns! < end,
  )
}

export function BatchTimeline() {
  const [{ state, journal }, store] = useStore()
  const [zoom, setZoom] = useState(8)
  const [asTable, setAsTable] = useState(false)
  const [follow, setFollow] = useState(true)
  const scroller = useRef<HTMLDivElement>(null)
  const s = state
  const sel = s.selection
  const cursorNs = s.cursor_ns ?? s.origin_ns ?? 0
  const started = s.batch_order.map((id) => s.batches[id]).filter((b) => b.started_ns !== null)
  const requests = s.request_order.map((id) => s.requests[id])

  const x = (ns: number) => PAD + relMs(s, ns) * zoom
  const width = x(cursorNs) + 80
  const height = AXIS + LANE * (1 + requests.length)
  const tick = TICKS.find((t) => t * zoom >= 60) ?? 10000
  // The SVG spans the whole run, but only ticks and bars within one screen of the viewport are rendered:
  // a live run can be minutes long at 8 px/ms, and drawing all of it every frame starves the renderer.
  const [scroll, setScroll] = useState({ left: 0, width: 1200 })
  const leftPx = follow ? Math.max(0, x(cursorNs) - scroll.width + 80) : scroll.left
  const lo = (leftPx - PAD - scroll.width) / zoom
  const hi = (leftPx - PAD + 2 * scroll.width) / zoom
  const visible = started.filter((b) => relMs(s, b.started_ns!) <= hi && relMs(s, b.completed_ns ?? cursorNs) >= lo)
  const firstTick = Math.max(0, Math.floor(lo / tick))
  const lastTick = Math.floor((Math.min(relMs(s, cursorNs), hi) / tick)) + 1
  const ticks = Array.from({ length: Math.max(0, lastTick - firstTick + 1) }, (_, i) => (firstTick + i) * tick)
  useEffect(() => {
    const el = scroller.current
    if (el) setScroll({ left: el.scrollLeft, width: el.clientWidth })
  }, [asTable])

  // Follow latest keeps the shared cursor in view (live tail, replay playhead, or a seek).
  useEffect(() => {
    const el = scroller.current
    if (follow && el) el.scrollLeft = Math.max(0, x(cursorNs) - el.clientWidth + 80)
  })

  function pick(b: BatchState) {
    store.select({ batch_id: b.batch_id, physical_block: null })
    // Land after the batch's last event (its commits and samples), not mid-range.
    const last = journal.findLast((e) => e.batch_id === b.batch_id)
    store.seek(last?.seq ?? s.high_watermark_seq)
  }

  function bar(b: BatchState, lane: number, mine: boolean) {
    const x0 = x(b.started_ns!)
    const w = Math.max(3, x(b.completed_ns ?? cursorNs) - x0)
    const y = AXIS + lane * LANE + 5
    const ongoing = b.completed_ns === null
    const label = `${b.phase === 'prefill' ? 'P' : 'D'}${lane === 0 ? ` ${b.batch_id}` : ''}${ongoing ? ' · ongoing' : ''}`
    return (
      <g
        key={`${b.batch_id}-${lane}`}
        className={`bar bar-${b.phase} ${mine ? 'is-mine' : ''} ${sel.batch_id === b.batch_id ? 'is-picked' : ''} ${ongoing ? 'is-ongoing' : ''} ${b.status === 'failed' ? 'is-failed' : ''}`}
        role="button"
        tabIndex={0}
        aria-label={describe(s, b)}
        onClick={() => pick(b)}
        onKeyDown={(e) => e.key === 'Enter' && pick(b)}
      >
        <title>{describe(s, b)}</title>
        <rect x={x0} y={y} width={w} height={LANE - 10} rx={2} />
        {w > label.length * 7 + 6 && (
          <text x={x0 + w / 2} y={y + (LANE - 10) / 2 + 4} textAnchor="middle">
            {label}
          </text>
        )}
      </g>
    )
  }

  return (
    <section className="pane timeline" aria-label="Continuous batching timeline">
      <div className="pane-head">
        <h2>Continuous batching</h2>
        <span className="muted">Host-observed engine intervals · new requests enter between engine steps</span>
        <span className="spacer" />
        <span className="legend">
          <span>
            <i className="sw sw-prefill" /> Prefill
          </span>
          <span>
            <i className="sw sw-decode" /> Decode
          </span>
          <span>
            <i className="sw sw-wait" /> Waiting
          </span>
          <span>
            <i className="sw sw-idle" /> Idle (other forward)
          </span>
        </span>
        <span className="sep" />
        <label className="follow">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow latest
        </label>
        <button className="btn btn-small" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>
          {asTable ? 'Timeline' : 'Table'}
        </button>
        <span className="label">Zoom</span>
        <button className="btn btn-small" aria-label="Zoom out" disabled={zoom === ZOOMS[0]} onClick={() => setZoom(ZOOMS[ZOOMS.indexOf(zoom) - 1])}>
          −
        </button>
        <button className="btn btn-small" aria-label="Zoom in" disabled={zoom === ZOOMS.at(-1)} onClick={() => setZoom(ZOOMS[ZOOMS.indexOf(zoom) + 1])}>
          +
        </button>
      </div>

      {asTable ? (
        <BatchTable state={s} batches={s.batch_order.map((id) => s.batches[id])} onPick={pick} />
      ) : (
        <div className="tl-body">
          <div className="tl-labels" style={{ paddingTop: AXIS }}>
            <div className="tl-label muted" style={{ height: LANE }}>
              Engine
            </div>
            {requests.map((r) => (
              <button
                key={r.request_id}
                className={`tl-label mono ${sel.request_id === r.request_id ? 'is-mine' : ''}`}
                style={{ height: LANE }}
                onClick={() => store.select({ request_id: r.request_id, batch_id: null, physical_block: null })}
              >
                <span className="tl-id" title={r.request_id}>{r.request_id}</span> <span className={`badge phase-${r.status}`}>{PHASE_LABEL[r.status]}</span>
              </button>
            ))}
          </div>
          <div
            className="tl-scroll"
            ref={scroller}
            onScroll={(e) => setScroll({ left: e.currentTarget.scrollLeft, width: e.currentTarget.clientWidth })}
          >
            <svg width={width} height={height} role="img" aria-label="Timeline in elapsed milliseconds">
              <defs>
                {['', '-mine'].map((v) => (
                  <pattern key={v} id={`tl-hatch${v}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line className={`hatch-line${v}`} x1="0" y1="0" x2="0" y2="6" />
                  </pattern>
                ))}
              </defs>
              {ticks.map((t) => (
                <g key={t} className="tick">
                  <line x1={PAD + t * zoom} x2={PAD + t * zoom} y1={AXIS - 6} y2={height} />
                  <text x={PAD + t * zoom + 3} y={AXIS - 10}>
                    {t} ms
                  </text>
                </g>
              ))}
              {[0, ...requests.map((_, i) => i + 1)].map((lane) => (
                <line key={lane} className="lane-line" x1={0} x2={width} y1={AXIS + lane * LANE + LANE / 2} y2={AXIS + lane * LANE + LANE / 2} />
              ))}

              {visible.map((b) => bar(b, 0, b.request_ids.includes(sel.request_id ?? '')))}

              {requests.map((r, i) => {
                const lane = i + 1
                const mid = AXIS + lane * LANE + LANE / 2
                const mine = r.request_id === sel.request_id
                const waitEnd = r.admitted_ns ?? r.terminal_ns ?? cursorNs
                return (
                  <g key={r.request_id}>
                    {r.queued_ns !== null && (
                      <g className="wait">
                        <title>{`${r.request_id} waiting from ${fmtMs(s, r.queued_ns)}${r.admitted_ns ? ` to ${fmtMs(s, r.admitted_ns)}` : ' (ongoing)'}${r.wait_reason ? ` · ${WAIT_LABEL[r.wait_reason]}` : ''}`}</title>
                        <line x1={x(r.queued_ns)} x2={Math.max(x(waitEnd), x(r.queued_ns) + 3)} y1={mid} y2={mid} />
                        {r.wait_reason && x(waitEnd) - x(r.queued_ns) > 90 && (
                          <text x={x(r.queued_ns) + 4} y={mid - 4}>
                            waiting · {WAIT_LABEL[r.wait_reason]}
                          </text>
                        )}
                      </g>
                    )}
                    {idleIntervals(s, r, started, visible).map((b) => {
                      const x0 = x(b.started_ns!)
                      const w = Math.max(3, x(b.completed_ns ?? cursorNs) - x0)
                      const text = `idle · ${b.batch_id} ${b.phase} ${b.request_ids.join(' ')}`
                      return (
                        <g key={`idle-${b.batch_id}`} className="idle">
                          <title>{`${r.request_id} keeps its cache; its forward is paused during ${text.slice(7)}`}</title>
                          <rect x={x0} y={AXIS + lane * LANE + 7} width={w} height={LANE - 14} rx={2} />
                          {w > 60 && (
                            <text x={x0 + w / 2} y={mid + 4} textAnchor="middle">
                              {w > text.length * 6 ? text : `idle · ${b.batch_id}`}
                            </text>
                          )}
                        </g>
                      )
                    })}
                    {visible.filter((b) => b.request_ids.includes(r.request_id)).map((b) => bar(b, lane, mine))}
                    {r.terminal_ns !== null && (
                      <g className={`terminal ${r.status}`}>
                        <title>{`${r.request_id} ${r.status} at ${fmtMs(s, r.terminal_ns)} · ${r.stop_reason}`}</title>
                        <line x1={x(r.terminal_ns)} x2={x(r.terminal_ns)} y1={mid - 9} y2={mid + 9} />
                        <text x={x(r.terminal_ns) + 4} y={mid + 4}>
                          {r.status === 'completed' ? '✓' : '✕'}
                        </text>
                      </g>
                    )}
                  </g>
                )
              })}

              <g className="cursor">
                <line x1={x(cursorNs)} x2={x(cursorNs)} y1={AXIS - 4} y2={height} />
                <text x={x(cursorNs) + 4} y={height - 4}>
                  seq {s.high_watermark_seq}
                </text>
              </g>
            </svg>
          </div>
        </div>
      )}
      <div className="tl-foot muted">Elapsed time since the run's first event (ms) · zoom preserves duration ratios</div>
    </section>
  )
}

function BatchTable({ state, batches, onPick }: { state: State; batches: BatchState[]; onPick: (b: BatchState) => void }) {
  if (batches.length === 0) return <p className="empty">No batches at this cursor.</p>
  return (
    <div className="tl-table">
      <table>
        <thead>
          <tr>
            {['Batch', 'Phase', 'Members (positions)', 'Start', 'End', 'Host', 'GPU', 'Status'].map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {batches.map((b) => (
            <tr key={b.batch_id} tabIndex={0} className={state.selection.batch_id === b.batch_id ? 'is-focused' : ''} onClick={() => onPick(b)} onKeyDown={(e) => e.key === 'Enter' && onPick(b)}>
              <td className="mono">{b.batch_id}</td>
              <td>{b.phase}</td>
              <td className="mono">
                {b.members.map((m) => `${m.request_id} ${m.start_position}–${m.start_position + m.input_tokens - 1}`).join(', ')}
              </td>
              <td className="mono">{fmtMs(state, b.started_ns)}</td>
              <td className="mono">{b.completed_ns === null ? (b.started_ns === null ? '—' : 'ongoing') : fmtMs(state, b.completed_ns)}</td>
              <td className="mono">{b.host_duration_ms === null ? '—' : `${b.host_duration_ms} ms`}</td>
              <td className="mono">{b.gpu_duration_ms === null ? 'unavailable' : `${b.gpu_duration_ms} ms`}</td>
              <td>{b.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
