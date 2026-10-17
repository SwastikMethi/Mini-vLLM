import { useStore } from '../state/store'
import { counts, fmtMs } from './format'
import { PresentationControls } from './PresentationControls'

export function StatusStrip() {
  const [{ state, live, cursor, source, playing, connection }] = useStore()
  const { active, waiting } = counts(state)
  const behind = live.high_watermark_seq - state.high_watermark_seq
  const mode =
    source === 'fixture'
      ? playing
        ? 'Fixture replay · playing'
        : 'Fixture replay'
      : cursor !== null
        ? 'Paused view · historical'
        : connection === 'open'
          ? 'Live inference'
          : connection === 'stale'
            ? 'Not connected · last synchronized state'
            : 'Connecting…'

  return (
    <div className="status-strip" role="status">
      <span className={`dot ${source === 'live' && cursor === null && connection === 'open' ? 'dot-accent' : 'dot-muted'}`} aria-hidden />
      <span className="mode">{mode}</span>
      <span className="sep" />
      <span className="accent-text">{active} active</span>
      <span className="sep" />
      <span>{waiting} waiting</span>
      <span className="sep" />
      <span className="mono muted">
        seq {state.high_watermark_seq} / {live.high_watermark_seq} · {fmtMs(state, state.cursor_ns)}
        {behind > 0 && ` · ${behind} events behind ${source === 'fixture' ? 'end' : 'live'}`}
      </span>
      {state.resync_needed && <span className="badge badge-error">Sequence gap · resync needed</span>}
      <span className="spacer" />
      <PresentationControls />
    </div>
  )
}
