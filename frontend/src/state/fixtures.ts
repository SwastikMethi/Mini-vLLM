import type { EngineEvent } from '../types/events'

export function parseJsonl(text: string): EngineEvent[] {
  return text
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as EngineEvent)
}
