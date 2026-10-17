import { cacheSummary, physicalBlocks, type BatchState, type State } from '../state/reducer'
import { useStore } from '../state/store'
import { batchRange, pid, plannedRange, within } from './format'

export function KVBlockGrid() {
  const [{ state }, store] = useStore()
  const { block_size, pool_blocks } = state.cache
  const blocks = physicalBlocks(state)
  const summary = cacheSummary(state)
  const { request_id: selReq, physical_block: selBlock, batch_id: selBatch } = state.selection
  const batch = selBatch ? state.batches[selBatch] : undefined
  const focus = selBlock !== null ? blocks[selBlock] : undefined

  return (
    <div className="kv-grid-wrap">
      <div className="pane-head">
        <h2>KV cache</h2>
        <span className="accent-text mono">
          {summary.owned_blocks} / {pool_blocks}
        </span>
        <span className="muted">blocks owned</span>
        <span className="spacer" />
        <span className="label">Physical blocks · {block_size} tokens / block</span>
      </div>

      {pool_blocks === 0 ? (
        <p className="empty">Waiting for engine.ready to report the pool.</p>
      ) : (
        <div className="kv-grid" role="list" aria-label="Physical KV blocks in ID order">
          {blocks.map((b) => {
            const mine = b.owner !== null && b.owner === selReq
            const r = b.owner ? state.requests[b.owner] : undefined
            const first = (b.logical_block ?? 0) * block_size
            const planned = r ? plannedRange(state, r.request_id) : null
            const writes = r && batchRange(batch, r.request_id)
            const touched = writes && writes[0] <= first + block_size - 1 && writes[1] >= first
            const plannedHere = planned && planned[0] <= first + block_size - 1 && planned[1] >= first
            const cls = [
              'tile',
              b.owner === null ? 'is-free' : mine ? 'is-mine' : 'is-other',
              b.owner !== null && b.valid_slots === 0 ? 'is-empty' : '',
              selBlock === b.physical_block ? 'is-focused' : '',
              touched ? 'is-batch' : '',
            ].join(' ')
            const state_text = b.owner === null ? 'free' : b.valid_slots === 0 ? 'owned, empty' : 'committed'
            return (
              <button
                key={b.physical_block}
                role="listitem"
                className={cls}
                aria-pressed={selBlock === b.physical_block}
                aria-label={`${pid(b.physical_block)} ${b.owner ? `owned by ${b.owner}, logical block ${b.logical_block}, ${b.valid_slots} of ${block_size} slots committed` : 'free'}`}
                onClick={() =>
                  store.select({ physical_block: b.physical_block, ...(b.owner ? { request_id: b.owner } : {}) })
                }
              >
                <span className="tile-id mono">{pid(b.physical_block)}</span>
                <span className="tile-owner mono">{b.owner ?? 'free'}</span>
                <span className="tile-bar" aria-hidden>
                  <span style={{ width: `${(b.valid_slots / block_size) * 100}%` }} />
                </span>
                <span className="tile-meta mono">
                  {b.owner === null ? '—' : `${b.valid_slots}/${block_size}`}
                  <span className="muted"> {plannedHere ? 'writing' : state_text}</span>
                </span>
              </button>
            )
          })}
        </div>
      )}

      {focus && <SlotStrip state={state} block={focus} batch={batch} />}

      <div className="legend">
        <span>
          <i className="sw sw-mine" /> Selected{selReq ? ` (${selReq})` : ''}
        </span>
        <span>
          <i className="sw sw-other" /> Allocated (other)
        </span>
        <span>
          <i className="sw sw-empty" /> Owned, empty
        </span>
        <span>
          <i className="sw sw-free" /> Free
        </span>
      </div>

      <div className="capacity mono">
        <span>Owned {summary.owned_blocks}</span>
        <span>Free {summary.free_blocks.length}</span>
        <span>Unused owned slots {summary.unused_owned_slots}</span>
        <span>
          Reserved credits {summary.reserved_capacity_blocks} / {pool_blocks}
        </span>
      </div>

    </div>
  )
}

function SlotStrip({
  state,
  block,
  batch,
}: {
  state: State
  block: ReturnType<typeof physicalBlocks>[number]
  batch: BatchState | undefined
}) {
  const { block_size } = state.cache
  const r = block.owner ? state.requests[block.owner] : undefined
  if (!r || block.logical_block === null) {
    return (
      <div className="slots-head mono">
        {pid(block.physical_block)} · free · no owner at seq {state.high_watermark_seq}
      </div>
    )
  }
  const first = block.logical_block * block_size
  const planned = plannedRange(state, r.request_id)
  const writes = batchRange(batch, r.request_id)
  return (
    <div className="slots">
      <div className="slots-head mono">
        {pid(block.physical_block)} · owner {r.request_id} · L{block.logical_block} · positions {first}–
        {first + block_size - 1} · {block.valid_slots} / {block_size} committed
      </div>
      <div className="slot-row" role="list" aria-label="Token slots">
        {Array.from({ length: block_size }, (_, k) => {
          const pos = first + k
          const kind = pos < r.cached_tokens ? 'committed' : within(pos, planned) ? 'planned' : 'empty'
          return (
            <span
              key={k}
              role="listitem"
              className={`slot slot-${kind} ${within(pos, writes) ? 'slot-batch' : ''}`}
              title={`position ${pos} · ${pid(block.physical_block)} offset ${k} · ${kind === 'planned' ? 'planned write (uncommitted)' : kind}`}
            />
          )
        })}
      </div>
    </div>
  )
}
