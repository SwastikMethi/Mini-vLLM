import { SPEEDS, useStore } from '../state/store'

export function PresentationControls() {
  const [{ live, cursor, playing, speed, source, base_seq }, store] = useStore()
  const latest = live.high_watermark_seq
  const shown = cursor ?? latest
  const atLatest = shown >= latest

  return (
    <div className="controls">
      {playing ? (
        <button className="btn" onClick={store.pause}>
          ❚❚ Pause
        </button>
      ) : cursor === null && source === 'live' ? (
        <button className="btn" onClick={store.pause}>
          ❚❚ Pause view
        </button>
      ) : (
        <button className="btn" onClick={store.play}>
          ▶ {atLatest ? 'Replay' : 'Play'}
        </button>
      )}
      <button className="btn" onClick={store.step} disabled={cursor === null || atLatest}>
        Step →
      </button>
      <label className="speed">
        <span className="sr-only">Playback speed</span>
        <select value={speed} onChange={(e) => store.setSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
      <input
        type="range"
        className="seek"
        aria-label="Event cursor"
        min={base_seq}
        max={latest}
        value={shown}
        onChange={(e) => store.seek(Number(e.target.value))}
      />
      <button className="btn" onClick={store.returnToLive} disabled={cursor === null}>
        {source === 'fixture' ? 'Jump to end' : 'Return to live'}
      </button>
    </div>
  )
}
