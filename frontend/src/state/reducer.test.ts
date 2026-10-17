import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import type { EngineEvent } from '../types/events'
import {
  applyEvent,
  applySnapshot,
  initialState,
  stateAt,
  toSnapshot,
  validSlots,
  type Snapshot,
  type State,
} from './reducer'
import { parseJsonl } from './fixtures'

const contracts = new URL('../../../mini_vllm_docs/contracts/', import.meta.url)
const read = (name: string) => readFileSync(new URL(name, contracts), 'utf8')
const json = (name: string) => JSON.parse(read(name)) as Snapshot

const single = parseJsonl(read('event_examples.jsonl'))
const batching = parseJsonl(read('continuous_batching_example.jsonl'))
const singleTerminal = json('snapshot_example.json')
const checkpoint = json('continuous_batching_checkpoint.json')
const batchingTerminal = json('continuous_batching_snapshot.json')

const replay = (events: readonly EngineEvent[], base?: State) => stateAt(events, Infinity, base)
const ms = (n: number) => n * 1_000_000

/** The snapshot fields the reducer can derive from events. */
function expectMatchesSnapshot(state: State, snap: Snapshot) {
  // trace_complete, fixture, metrics and manifest.trace_mode are server-side, not event-derived.
  const { trace_complete: _t, fixture: _f, metrics: _m, manifest, ...rest } = snap
  const { trace_mode: _tm, ...manifestFromEvents } = manifest
  expect(state.resync_needed).toBe(false)
  expect(toSnapshot(state)).toMatchObject({ ...rest, manifest: manifestFromEvents })
}

/** Every intermediate state of a journal, indexed by seq. */
function everyState(events: readonly EngineEvent[]) {
  const states: State[] = []
  let s = initialState()
  for (const e of events) states.push((s = applyEvent(s, e)))
  return states
}

describe('fixture replay', () => {
  it('event_examples.jsonl reproduces snapshot_example.json', () => {
    expectMatchesSnapshot(replay(single), singleTerminal)
  })

  it('continuous_batching_example.jsonl at its 69 ms cursor reproduces the checkpoint', () => {
    const s = stateAt(batching, checkpoint.high_watermark_seq)
    expect(s.cursor_ns).toBe(ms(69))
    expectMatchesSnapshot(s, checkpoint)
  })

  it('continuous_batching_example.jsonl reproduces the terminal snapshot', () => {
    expectMatchesSnapshot(replay(batching), batchingTerminal)
  })
})

describe('ordering', () => {
  it('drops duplicate seq without changing state', () => {
    const s = stateAt(batching, 20)
    expect(applyEvent(s, batching[19])).toBe(s)
    expect(applyEvent(s, batching[3])).toBe(s)
  })

  it('flags a gap and applies nothing until a snapshot', () => {
    const s = stateAt(batching, 49)
    const gapped = applyEvent(s, batching[50]) // seq 51 after 49
    expect(gapped.resync_needed).toBe(true)
    expect(gapped.high_watermark_seq).toBe(49)
    expect(applyEvent(gapped, batching[49])).toBe(gapped)
    expectMatchesSnapshot(applySnapshot(batchingTerminal), batchingTerminal)
  })

  it('flags an event from another run', () => {
    const s = stateAt(batching, 5)
    expect(applyEvent(s, { ...batching[5], run_id: 'other' }).resync_needed).toBe(true)
  })

  it('stateAt rebuilds any historical seq identically to incremental application', () => {
    const states = everyState(batching)
    for (const e of batching) expect(stateAt(batching, e.seq)).toEqual(states[e.seq - 1])
  })
})

describe('projection rules', () => {
  it('only prefill/decode completion advances cached_tokens', () => {
    const states = everyState(batching)
    batching.forEach((e, i) => {
      if (i === 0 || e.type === 'request.prefill.completed' || e.type === 'request.decode.completed') return
      if (e.type === 'cache.released') return
      for (const id of states[i - 1].request_order) {
        expect(states[i].requests[id]?.cached_tokens).toBe(states[i - 1].requests[id].cached_tokens)
      }
    })
  })

  it('executing_batch_id is null between forwards even while decoding', () => {
    const a = stateAt(batching, 31).requests.A // A0/A1 sampled, B prefilled, b04 not started
    expect(a.status).toBe('decoding')
    expect(a.executing_batch_id).toBeNull()
    expect(stateAt(batching, 36).requests.A.executing_batch_id).toBe('b04')
    expect(stateAt(batching, 37).requests.A.executing_batch_id).toBeNull()
  })

  it('cache.released clears ownership, slots and credits but keeps processed_tokens', () => {
    const s = stateAt(batching, 55)
    expect(s.requests.A).toMatchObject({ block_table: [], cached_tokens: 0, reservation_blocks: 0, processed_tokens: 18 })
    expect(Object.values(s.cache.owners)).not.toContain('A')
  })

  it('a decode commit at a block boundary fills exactly one slot in a newly appended block', () => {
    // event_examples: 16 cached in P7, then growth allocates P2 (seq 20), decode commits position 16 (seq 25).
    const grown = stateAt(single, 20).requests['example-request-A']
    expect(grown.block_table).toEqual([7, 2])
    expect(validSlots(grown, 1, 16)).toBe(0) // allocated, not committed
    const committed = stateAt(single, 25).requests['example-request-A']
    expect(validSlots(committed, 0, 16)).toBe(16)
    expect(validSlots(committed, 1, 16)).toBe(1)
    expect(stateAt(single, 26).requests['example-request-A'].cached_tokens).toBe(17) // token.sampled adds nothing
  })
})

describe('ui_projection_contract acceptance checks', () => {
  it('token sampling alone never increments committed slots', () => {
    const states = everyState([...single])
    const batchingStates = everyState(batching)
    for (const [events, ss] of [[single, states], [batching, batchingStates]] as const) {
      events.forEach((e, i) => {
        if (e.type !== 'token.sampled') return
        expect(ss[i].requests[e.request_id].cached_tokens).toBe(ss[i - 1].requests[e.request_id].cached_tokens)
        expect(ss[i].cache).toEqual(ss[i - 1].cache)
      })
    }
  })

  it("a newcomer can prefill while an older request stays decoding with an idle lane", () => {
    const s = stateAt(batching, 28) // b03 started
    expect(s.batches.b03).toMatchObject({ status: 'active', phase: 'prefill', request_ids: ['B'] })
    expect(s.requests.B.executing_batch_id).toBe('b03')
    expect(s.requests.A.status).toBe('decoding')
    expect(s.requests.A.executing_batch_id).toBeNull()
  })

  it('batch membership, input positions and block mappings agree at every cursor', () => {
    for (const s of everyState(batching)) {
      const bs = s.cache.block_size
      for (const id of s.request_order) {
        const r = s.requests[id]
        r.block_table.forEach((p) => expect(s.cache.owners[p]).toBe(id))
        expect(r.cached_tokens).toBeLessThanOrEqual(r.block_table.length * bs)
      }
      for (const [p, owner] of Object.entries(s.cache.owners)) {
        expect(s.requests[owner].block_table).toContain(Number(p))
      }
      for (const id of s.batch_order) {
        const b = s.batches[id]
        if (b.status !== 'active') continue
        for (const m of b.members) {
          const r = s.requests[m.request_id]
          expect(r.executing_batch_id).toBe(id)
          expect(m.start_position).toBe(r.cached_tokens) // writes start at the next uncommitted position
          const lastBlock = Math.floor((m.start_position + m.input_tokens - 1) / bs)
          expect(r.block_table[lastBlock]).toBeDefined() // every planned write has a physical block
        }
      }
    }
  })

  it('a queue update retains the first queued timestamp', () => {
    const before = stateAt(batching, 45).requests.C
    const after = stateAt(batching, 59).requests.C // re-queued at 86 ms with a new reason
    expect(after.wait_reason).toBe('awaiting_scheduler')
    expect(after.queued_ns).toBe(before.queued_ns)
    expect(after.queued_ns).toBe(ms(69))
  })

  it('completed batches never overlap in the single engine lane', () => {
    const s = replay(batching)
    const done = s.batch_order.map((id) => s.batches[id]).filter((b) => b.status === 'completed')
    done.sort((x, y) => x.started_ns! - y.started_ns!)
    for (let i = 1; i < done.length; i++) expect(done[i].started_ns!).toBeGreaterThanOrEqual(done[i - 1].completed_ns!)
  })

  it('reconnecting from the checkpoint and replaying reproduces the terminal snapshot', () => {
    const reconnected = applySnapshot(checkpoint)
    expectMatchesSnapshot(reconnected, checkpoint)
    // Resume from a few events before the watermark: overlap is ignored as duplicates.
    expectMatchesSnapshot(replay(batching.slice(40), reconnected), batchingTerminal)
  })

  it('a reused physical block keeps its historical owner in replay', () => {
    const live = replay(batching.slice(0, 70)) // C owns P3 after decode growth at seq 68
    expect(live.cache.owners[3]).toBe('C')
    expect(stateAt(batching, 45).cache.owners[3]).toBe('A')
    expect(live.cache.owners[3]).toBe('C') // rebuilding history never touched the live state
  })
})
