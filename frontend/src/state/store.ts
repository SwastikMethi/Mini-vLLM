// External store around the reducer. Holds the accepted journal, the live state
// and the shared presentation cursor; React reads it via useSyncExternalStore.
// Events apply immediately; subscribers are notified at most once per frame.
import { createContext, useContext, useSyncExternalStore } from 'react'
import type { EngineEvent } from '../types/events'
import { applyEvent, applySnapshot, initialState, reduce, stateAt, type Selection, type Snapshot, type State } from './reducer'

export type Source = 'fixture' | 'live'
export type Connection = 'connecting' | 'open' | 'stale' | 'none'
export interface EngineHealth {
  status: 'loading' | 'ready' | 'failed'
  error: string | null
  model_id: string
  device: string
}
export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const

export interface View {
  /** State at the displayed cursor; every panel renders this. */
  state: State
  /** Newest applied state. */
  live: State
  journal: readonly EngineEvent[]
  /** First seq the journal can rebuild (0 for a full run, the watermark after a snapshot). */
  base_seq: number
  /** Displayed seq, or null when following the newest event. */
  cursor: number | null
  playing: boolean
  speed: number
  source: Source
  connection: Connection
  /** Backend health before engine.ready arrives (live mode only). */
  health: EngineHealth | null
}

const raf: (cb: FrameRequestCallback) => unknown =
  globalThis.requestAnimationFrame?.bind(globalThis) ?? ((cb) => setTimeout(() => cb(performance.now()), 16))

export function createStore(source: Source) {
  let base = initialState()
  let journal: EngineEvent[] = []
  let live = base
  let cursor: number | null = null
  let playing = false
  let speed = 1
  let connection: Connection = source === 'live' ? 'connecting' : 'none'
  let health: EngineHealth | null = null
  // Event receipt → render timing (performance.now ms), for the p95 event-to-render measurement.
  let pendingReceipt: number | null = null
  const renderDelays: number[] = []
  // Last historical rebuild, so stepping forward folds incrementally.
  let shown: State | null = null

  const listeners = new Set<() => void>()
  let published: View = snapshot()
  let frameQueued = false

  function snapshot(): View {
    return {
      state: displayed(), live, journal, base_seq: base.high_watermark_seq, cursor, playing, speed, source, connection, health,
    }
  }

  function displayed(): State {
    if (cursor === null) return live
    if (!shown || shown.high_watermark_seq > cursor) shown = stateAt(journal, cursor, base)
    else if (shown.high_watermark_seq < cursor) shown = stateAt(journal, cursor, shown)
    return { ...shown, selection: live.selection }
  }

  function changed() {
    if (frameQueued) return
    frameQueued = true
    raf(() => {
      frameQueued = false
      published = snapshot()
      listeners.forEach((l) => l())
      if (pendingReceipt !== null) {
        const received = pendingReceipt
        pendingReceipt = null
        // Measured after React commits this frame's update (next macrotask after the synchronous render).
        setTimeout(() => renderDelays.push(performance.now() - received))
      }
    })
  }

  const eventAt = (seq: number) => journal[seq - base.high_watermark_seq - 1]

  function setCursor(seq: number | null) {
    cursor = seq === null ? null : Math.max(base.high_watermark_seq, Math.min(seq, live.high_watermark_seq))
    changed()
  }

  let lastFrame = 0
  let playhead = 0
  function tick(now: number) {
    if (!playing || cursor === null) return
    playhead += (now - lastFrame) * 1e6 * speed
    lastFrame = now
    let c = cursor
    // Skip recorded idle gaps (no live request at the cursor); spacing during active work is never compressed.
    const next = eventAt(c + 1)
    if (next && next.monotonic_ns > playhead && Object.values(displayed().requests).every((r) => r.terminal_ns !== null))
      playhead = next.monotonic_ns
    while (c < live.high_watermark_seq && eventAt(c + 1).monotonic_ns <= playhead) c++
    if (c >= live.high_watermark_seq) {
      playing = false
      setCursor(source === 'live' ? null : c)
      return
    }
    setCursor(c)
    raf(tick)
  }

  const store = {
    subscribe(l: () => void) {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    getSnapshot: () => published,
    /** Newest applied state, without waiting for the next painted frame (for the network client). */
    current: () => live,

    setConnection(c: Connection) {
      if (c === connection) return
      connection = c
      changed()
    },
    setEngine(h: EngineHealth) {
      health = h
      changed()
    },
    /** Records when the oldest not-yet-rendered event arrived. */
    markReceived(at: number) {
      pendingReceipt ??= at
    },
    /** Event-to-render delays in ms (oldest pending event per painted frame). */
    renderDelays: () => renderDelays as readonly number[],

    /** Applies engine events in order; duplicates are dropped, a gap sets resync_needed. */
    ingest(events: readonly EngineEvent[]) {
      for (const e of events) {
        const next = applyEvent(live, e)
        if (next.high_watermark_seq > live.high_watermark_seq) journal = [...journal, e]
        live = next
      }
      changed()
    },
    /** Replaces everything with an atomic snapshot; history before its watermark is unavailable. */
    loadSnapshot(snap: Snapshot) {
      base = live = applySnapshot(snap, live.selection)
      journal = []
      shown = null
      cursor = null
      playing = false
      changed()
    },
    select(selection: Partial<Selection>) {
      live = reduce(live, { kind: 'select', selection })
      changed()
    },
    /** Moves every panel to a recorded seq (historical view). */
    seek(seq: number) {
      playing = false
      setCursor(seq)
    },
    /** Advances one recorded event; no-op at the newest event. */
    step() {
      playing = false
      if (cursor !== null && cursor < live.high_watermark_seq) setCursor(cursor + 1)
    },
    /** Freezes presentation at the newest event; ingestion continues. */
    pause() {
      playing = false
      setCursor(cursor ?? live.high_watermark_seq)
    },
    /** Plays recorded events from the cursor at their original spacing, scaled by speed. */
    play() {
      if (cursor === null && source === 'live') return
      if ((cursor ?? Infinity) >= live.high_watermark_seq) setCursor(base.high_watermark_seq) // replay from start
      playing = true
      playhead = eventAt(cursor!)?.monotonic_ns ?? journal[0]?.monotonic_ns ?? 0
      raf((now) => {
        lastFrame = now
        raf(tick)
      })
      changed()
    },
    setSpeed(s: number) {
      speed = s
      changed()
    },
    /** Follows the newest event again. */
    returnToLive() {
      playing = false
      setCursor(null)
    },
  }
  return store
}

export type Store = ReturnType<typeof createStore>

export const StoreContext = createContext<Store | null>(null)

export function useStore(): [View, Store] {
  const store = useContext(StoreContext)
  if (!store) throw new Error('StoreContext missing')
  return [useSyncExternalStore(store.subscribe, store.getSnapshot), store]
}
