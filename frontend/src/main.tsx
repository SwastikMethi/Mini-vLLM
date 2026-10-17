import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import batchingJsonl from '@contracts/continuous_batching_example.jsonl?raw'
import singleJsonl from '@contracts/event_examples.jsonl?raw'
import { connect, stream } from './api/client'
import { AppShell } from './components/AppShell'
import { parseJsonl } from './state/fixtures'
import { createStore, StoreContext } from './state/store'
import './theme.css'
import './app.css'

// Live mode is the default route; ?fixture=<name> loads a labelled illustrative trace with no backend.
const FIXTURES = {
  batching: { label: 'Continuous batching', jsonl: batchingJsonl, startSeq: 45 },
  single: { label: 'Single request', jsonl: singleJsonl, startSeq: 20 },
}
type FixtureName = keyof typeof FIXTURES

const param = new URLSearchParams(location.search).get('fixture')
const name: FixtureName = param && param in FIXTURES ? (param as FixtureName) : 'batching'
const store = createStore(param ? 'fixture' : 'live')

if (param) {
  const fixture = FIXTURES[name]
  const events = parseJsonl(fixture.jsonl)
  store.ingest(events)
  store.seek(fixture.startSeq)
  store.select({ request_id: events.find((e) => e.type === 'request.accepted')?.request_id ?? null })
} else {
  void connect(store)
  // Exposed for the M5 event-to-render measurement.
  Object.assign(window, { miniVllmStore: store, miniVllmStream: stream })
}
const options = Object.entries(FIXTURES).map(([value, f]) => ({ value, label: f.label }))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <StoreContext.Provider value={store}>
      <AppShell fixture={name} fixtures={options} />
    </StoreContext.Provider>
  </StrictMode>,
)
