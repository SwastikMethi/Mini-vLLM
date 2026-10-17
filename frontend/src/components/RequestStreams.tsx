import type { KeyboardEvent } from 'react'
import { useStore } from '../state/store'
import { RequestRow } from './RequestRow'

export function RequestStreams({ onInspect }: { onInspect: () => void }) {
  const [{ state }, store] = useStore()
  const ids = state.request_order
  const selected = state.selection.request_id
  const batch = state.selection.batch_id ? state.batches[state.selection.batch_id] : undefined

  function select(id: string) {
    store.select({ request_id: id, batch_id: null, physical_block: null })
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    const i = selected ? ids.indexOf(selected) : -1
    const next = ids[Math.max(0, Math.min(ids.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))]
    if (next) {
      select(next)
      document.getElementById(`row-${next}`)?.focus()
    }
  }

  return (
    <section className="pane streams" aria-label="Request streams">
      <div className="pane-head">
        <h2>Request streams</h2>
        <span className="muted mono">{ids.length} requests</span>
      </div>
      {ids.length === 0 ? (
        <p className="empty">No requests at this cursor. Send a prompt to see waiting, allocation, prefill and decode.</p>
      ) : (
        <ul className="rows" role="listbox" aria-label="Requests" onKeyDown={onKeyDown}>
          {ids.map((id) => (
            <RequestRow
              key={id}
              r={state.requests[id]}
              state={state}
              selected={id === selected}
              inBatch={batch?.request_ids.includes(id) ? batch.batch_id : null}
              onSelect={() => select(id)}
              onInspect={() => {
                select(id)
                onInspect()
              }}
            />
          ))}
        </ul>
      )}
      <p className="hint-line muted">Select a request to inspect its pages and timeline lane.</p>
    </section>
  )
}
