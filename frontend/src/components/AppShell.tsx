import { useEffect, useState } from 'react'
import { api, stream } from '../api/client'
import { useStore } from '../state/store'
import { BatchTimeline } from './BatchTimeline'
import { BlockTable } from './BlockTable'
import { DetailDrawer } from './DetailDrawer'
import { KVBlockGrid } from './KVBlockGrid'
import { RequestComposer } from './RequestComposer'
import { RequestStreams } from './RequestStreams'
import { StatusStrip } from './StatusStrip'

const EDITABLE = 'input, textarea, select, button, [contenteditable="true"]'

export function AppShell({ fixture, fixtures }: { fixture: string; fixtures: { value: string; label: string }[] }) {
  const [view, store] = useStore()
  const [inspecting, setInspecting] = useState(false)
  const { state, source, connection, health } = view
  const isFixture = source === 'fixture'
  const [submitError, setSubmitError] = useState<string | null>(null)
  const engineStatus = state.engine_status === 'loading' && health ? health.status : state.engine_status

  function submit(prompt: string, maxNewTokens: number) {
    setSubmitError(null)
    api
      .submit(prompt, maxNewTokens)
      .then(({ request_id }) => store.select({ request_id }))
      .catch((e: Error) => setSubmitError(e.message))
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setInspecting(false)
      if (e.key === ' ' && !(e.target as Element).closest?.(EDITABLE)) {
        e.preventDefault()
        if (view.playing || (source === 'live' && view.cursor === null)) store.pause()
        else store.play()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [store, source, view.playing, view.cursor])

  return (
    <div className="app">
      <header className="appbar">
        <span className="brand">
          <span className="brand-mark">&gt;_</span> mini-vllm
        </span>
        <nav className="tabs" aria-label="Views">
          <span className="tab is-active" aria-current="page">
            {isFixture ? 'Fixture' : 'Live'}
          </span>
        </nav>
        <span className="spacer" />
        {isFixture && (
          <label className="fixture-pick">
            <span className="label">Trace</span>
            <select
              value={fixture}
              onChange={(e) => (location.search = `?fixture=${e.target.value}`)}
            >
              {fixtures.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className="mono model">
          {state.manifest?.model_id ?? health?.model_id ?? 'model loading'}
          {state.manifest && <span className="muted"> · {state.manifest.device} {state.manifest.dtype}</span>}
        </span>
        {isFixture && <span className="badge badge-outline">Fixture · illustrative</span>}
      </header>

      <RequestComposer
        ready={engineStatus === 'ready' && connection === 'open'}
        disabledReason={
          isFixture
            ? 'Fixture replay: sending needs the live backend.'
            : engineStatus === 'failed'
              ? `Engine failed: ${state.engine_error?.message ?? health?.error ?? 'see server log'}`
              : undefined
        }
        error={submitError}
        onSubmit={isFixture ? undefined : submit}
      />
      <StatusStrip />

      <main className="panes">
        <RequestStreams onInspect={() => setInspecting(true)} />
        <section className="pane kv" aria-label="KV cache">
          <KVBlockGrid />
          <BlockTable />
        </section>
      </main>

      <BatchTimeline />

      <footer className="footer">
        <span
          className={`dot ${state.resync_needed || connection === 'stale' ? 'dot-error' : connection === 'open' ? 'dot-accent' : 'dot-muted'}`}
          aria-hidden
        />
        <span>
          {isFixture
            ? 'Fixture · no backend connection'
            : connection === 'open'
              ? `Engine ${engineStatus} · events synchronized`
              : connection === 'stale'
                ? `Connection lost · showing seq ${state.high_watermark_seq}, reconnecting…`
                : 'Connecting…'}
        </span>
        {connection === 'stale' && (
          <button className="btn btn-small" onClick={() => stream.reconnect()}>
            Reconnect
          </button>
        )}
        <span className="sep" />
        <span>
          Block size <span className="mono">{state.cache.block_size}</span>
        </span>
        <span className="sep" />
        <span>
          Pool <span className="mono">{state.cache.pool_blocks}</span> blocks
        </span>
        <span className="spacer" />
        <span className="hint">
          <kbd>↑↓</kbd> Select · <kbd>Enter</kbd> Inspect · <kbd>Esc</kbd> Close · <kbd>Space</kbd>{' '}
          {isFixture ? 'Play/pause' : 'Pause view'}
        </span>
      </footer>

      {inspecting && <DetailDrawer onClose={() => setInspecting(false)} />}
    </div>
  )
}
