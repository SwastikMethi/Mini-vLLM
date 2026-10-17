import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { parseJsonl } from './fixtures'
import { stateAt } from './reducer'
import { createStore } from './store'

const events = parseJsonl(
  readFileSync(new URL('../../../mini_vllm_docs/contracts/continuous_batching_example.jsonl', import.meta.url), 'utf8'),
)

afterEach(() => vi.useRealTimers())

describe('store', () => {
  it('notifies at most once per animation frame', () => {
    vi.useFakeTimers()
    const store = createStore('fixture')
    const listener = vi.fn()
    store.subscribe(listener)
    for (const e of events) store.ingest([e])
    store.select({ request_id: 'A' })
    expect(listener).not.toHaveBeenCalled()
    vi.advanceTimersByTime(16)
    expect(listener).toHaveBeenCalledTimes(1)
    expect(store.getSnapshot().live.high_watermark_seq).toBe(76)
  })

  it('seek, step and returnToLive share one cursor across all state', () => {
    vi.useFakeTimers()
    const store = createStore('fixture')
    store.ingest([...events, events[3]]) // trailing duplicate is dropped
    store.select({ request_id: 'C' })
    store.seek(45)
    store.step()
    vi.advanceTimersByTime(16)
    const v = store.getSnapshot()
    expect(v.journal).toHaveLength(76)
    expect(v.cursor).toBe(46)
    expect(v.state).toEqual({ ...stateAt(events, 46), selection: v.live.selection })
    expect(v.state.selection.request_id).toBe('C')
    store.returnToLive()
    vi.advanceTimersByTime(16)
    expect(store.getSnapshot().state).toBe(store.getSnapshot().live)
  })

  it('plays recorded spacing scaled by speed and ends on the live state', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'performance'] })
    const store = createStore('fixture')
    store.ingest(events)
    store.seek(2) // A accepted at 1 ms
    store.setSpeed(0.25)
    store.play()
    const recordedAt = () => events[store.getSnapshot().cursor! - 1].monotonic_ns / 1e6
    vi.advanceTimersByTime(100) // past the ~2-frame start latency
    const t1 = recordedAt()
    vi.advanceTimersByTime(200) // 200 ms wall at 0.25x ≈ 50 ms recorded
    const advanced = recordedAt() - t1
    expect(advanced).toBeGreaterThan(35) // bounds absorb frame quantization and event spacing
    expect(advanced).toBeLessThan(65)
    vi.advanceTimersByTime(1000)
    const v = store.getSnapshot()
    expect(v.playing).toBe(false)
    expect(v.state.requests).toEqual(v.live.requests)
  })

  it('skips recorded idle gaps but never compresses active work', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'performance'] })
    const gap = 10e9 // 10 s of idle engine before the first request
    const shifted = events.map((e) => (e.seq === 1 ? e : { ...e, monotonic_ns: e.monotonic_ns + gap }))
    const store = createStore('fixture')
    store.ingest(shifted)
    store.seek(1)
    store.play()
    vi.advanceTimersByTime(50)
    const c = store.getSnapshot().cursor!
    expect(c).toBeGreaterThan(2) // jumped the idle gap
    expect(shifted[c - 1].monotonic_ns - shifted[1].monotonic_ns).toBeLessThan(60e6) // then real spacing
  })
})
