import { useState, useEffect } from 'react'
import { openRunStream, openWatchStream } from '../api'
import type { LiveRunState, LiveSample, StreamEvent } from '../types'

export function useLiveRun(runId: string, initialStatus: string) {
  const [state, setState] = useState<LiveRunState>({
    run_id: runId,
    config_name: '',
    start_time: Date.now() / 1000,
    status: initialStatus as LiveRunState['status'],
    total: null,
    samples: new Map(),
    stdout_lines: [],
  })

  useEffect(() => {
    // Don't open SSE for finished runs
    if (initialStatus !== 'running' && initialStatus !== 'watching') return

    const es = initialStatus === 'watching' ? openWatchStream(runId) : openRunStream(runId)

    es.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data) as StreamEvent
        setState(prev => applyEvent(prev, data))
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      es.close()
      setState(prev => ({ ...prev, status: 'completed' }))
    }

    return () => es.close()
  }, [runId, initialStatus])

  return state
}

function applyEvent(prev: LiveRunState, event: StreamEvent): LiveRunState {
  switch (event.type) {
    case 'eval_start': {
      return { ...prev, total: event.total }
    }
    case 'sample_start': {
      const samples = new Map(prev.samples)
      const sample: LiveSample = { id: event.id, name: event.name, status: 'running', turn: 0, score: null, messages: [] }
      samples.set(event.id, sample)
      return { ...prev, samples }
    }
    case 'turn': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.id)
      if (existing) {
        samples.set(event.id, { ...existing, turn: event.turn, status: 'running' })
      }
      return { ...prev, samples }
    }
    case 'assistant_message': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.sample_id)
      if (existing) {
        const messages = [...existing.messages, { turn: event.turn, content: event.content, timestamp: event.timestamp }]
        samples.set(event.sample_id, { ...existing, messages })
      }
      return { ...prev, samples }
    }
    case 'sample_end': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.id)
      if (existing) {
        samples.set(event.id, { ...existing, status: 'done', score: event.score })
      }
      return { ...prev, samples }
    }
    case 'eval_end': {
      return { ...prev, status: 'completed' }
    }
    case 'complete': {
      const status = event.status === 'success' ? 'completed' : 'failed'
      return { ...prev, status }
    }
    case 'stdout': {
      const stdout_lines = [...prev.stdout_lines, event.line].slice(-200)
      return { ...prev, stdout_lines }
    }
    default:
      return prev
  }
}
