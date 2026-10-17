import { useState } from 'react'

interface Props {
  ready: boolean
  /** Set when sending is unavailable (fixture mode). */
  disabledReason?: string
  onSubmit?: (prompt: string, maxNewTokens: number) => void
  /** Last submission error from the backend (422/429/503 …). */
  error?: string | null
}

export function RequestComposer({ ready, disabledReason, onSubmit, error }: Props) {
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(128)
  const canSend = ready && !disabledReason && !!onSubmit && prompt.trim().length > 0

  function send() {
    if (!canSend) return
    onSubmit!(prompt, maxTokens)
    setPrompt('')
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        send()
      }}
    >
      <label className="label" htmlFor="prompt">
        New request
      </label>
      <div className="composer-row">
        <textarea
          id="prompt"
          className="mono"
          rows={1}
          placeholder={disabledReason ?? (ready ? 'Ask the model something…' : 'Model loading…')}
          value={prompt}
          disabled={!!disabledReason}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              send()
            }
          }}
        />
        <label className="max-tokens">
          <span>Max tokens</span>
          <input
            type="number"
            className="mono"
            min={1}
            max={256}
            value={maxTokens}
            disabled={!!disabledReason}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
          />
        </label>
        <button type="submit" className="btn btn-primary" disabled={!canSend} title={disabledReason}>
          Send ↵
        </button>
      </div>
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
    </form>
  )
}
