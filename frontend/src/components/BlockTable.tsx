import { validSlots } from '../state/reducer'
import { useStore } from '../state/store'
import { batchRange, fmtMs, pid } from './format'

export function BlockTable() {
  const [{ state }, store] = useStore()
  const { block_size } = state.cache
  const { request_id, physical_block, batch_id } = state.selection
  const r = request_id ? state.requests[request_id] : undefined
  const writes = r && batch_id ? batchRange(state.batches[batch_id], r.request_id) : null

  let empty: string | null = null
  if (!r) empty = 'Select a request to see its logical block table.'
  else if (r.block_table.length === 0)
    empty = r.release_reason
      ? `Pages released at ${fmtMs(state, r.terminal_ns ?? state.cursor_ns)}; ${r.processed_tokens} inputs processed (history).`
      : `No blocks owned (${r.status === 'queued' ? 'waiting' : r.status}).`

  return (
    <div className="block-table">
      <div className="pane-head">
        <h3 className="mono">{r?.request_id ?? '—'}</h3>
        <span className="label">/ Block table</span>
        <span className="spacer" />
        {r && (
          <span className="muted mono">
            {r.cached_tokens} cached · {r.reservation_blocks} credits
          </span>
        )}
      </div>
      {empty ? (
        <p className="empty">{empty}</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th scope="col">Logical block</th>
              <th scope="col">Token range</th>
              <th scope="col">Physical block</th>
              <th scope="col">Committed slots</th>
            </tr>
          </thead>
          <tbody>
            {r!.block_table.map((p, j) => {
              const first = j * block_size
              const last = first + block_size - 1
              const hit = writes && writes[0] <= last && writes[1] >= first
              return (
                <tr
                  key={j}
                  className={`${physical_block === p ? 'is-focused' : ''} ${hit ? 'is-batch' : ''}`}
                  onClick={() => store.select({ physical_block: p })}
                  tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && store.select({ physical_block: p })}
                >
                  <td className="mono">L{j}</td>
                  <td className="mono">
                    {first} – {last}
                  </td>
                  <td className="mono accent-text">{pid(p)}</td>
                  <td className="mono">
                    {validSlots(r!, j, block_size)} of {block_size}
                    {hit && (
                      <span className="muted">
                        {' '}
                        · {batch_id} writes {Math.max(writes[0], first)}–{Math.min(writes[1], last)}
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
