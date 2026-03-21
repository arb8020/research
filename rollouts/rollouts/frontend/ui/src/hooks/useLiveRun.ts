import { useState, useEffect } from 'react'
import { openRunEvents } from '../api'
import type { LiveRunState, LiveSample, StreamEvent, TraceSample } from '../types'

function makeInitialState(runId: string, initialStatus: string): LiveRunState {
  return {
    run_id: runId,
    config_name: '',
    start_time: Date.now() / 1000,
    status: initialStatus as LiveRunState['status'],
    total: null,
    samples: new Map(),
    stdout_lines: [],
  }
}

function buildInitialTraceSample(
  sampleId: string,
  sampleData?: Record<string, unknown> | null,
  messages?: TraceSample['trajectory']['messages'],
): TraceSample {
  return {
    id: sampleId,
    index: null,
    group_index: null,
    input: sampleData ?? null,
    prompt: typeof sampleData?.prompt === 'string' ? sampleData.prompt : '',
    ground_truth: typeof sampleData?.ground_truth === 'string' ? sampleData.ground_truth : null,
    trajectory: {
      completions: [],
      messages: messages ?? [],
    },
    metadata: {
      status: 'pending',
      turns_used: 0,
    },
  }
}

export function useLiveRun(runId: string, initialStatus: string) {
  const [state, setState] = useState<LiveRunState>(() => makeInitialState(runId, initialStatus))

  useEffect(() => {
    // Don't open SSE for finished runs
    if (initialStatus !== 'running' && initialStatus !== 'watching') return

    const es = openRunEvents(runId)

    es.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data) as StreamEvent
        setState(prev =>
          applyEvent(
            prev.run_id === runId ? prev : makeInitialState(runId, initialStatus),
            data,
          ),
        )
      } catch (err) {
        console.error('Failed to parse live run stream event', err)
      }
    }

    es.onerror = () => {
      es.close()
      setState(prev => ({
        ...(prev.run_id === runId ? prev : makeInitialState(runId, initialStatus)),
        status: 'completed',
      }))
    }

    return () => es.close()
  }, [runId, initialStatus])

  if (state.run_id !== runId) {
    return makeInitialState(runId, initialStatus)
  }

  return state
}

function applyEvent(prev: LiveRunState, event: StreamEvent): LiveRunState {
  switch (event.type) {
    case 'eval_start': {
      return { ...prev, total: event.total, config_name: event.name }
    }
    case 'sample_start': {
      const samples = new Map(prev.samples)
      const initialMessages = (event.messages ?? []).map(message => ({
        ...message,
        content: message.content ?? '',
      }))
      const sample: LiveSample = {
        id: event.id,
        name: event.name,
        status: 'running',
        turn: 0,
        score: null,
        messages: [],
        sample: buildInitialTraceSample(event.id, event.sample_data, initialMessages),
      }
      samples.set(event.id, sample)
      return { ...prev, samples }
    }
    case 'turn': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.id)
      if (existing) {
        samples.set(event.id, {
          ...existing,
          turn: event.turn,
          status: 'running',
          sample: {
            ...existing.sample,
            metadata: {
              ...existing.sample.metadata,
              turns_used: event.turn,
              status: event.status || 'running',
            },
          },
        })
      }
      return { ...prev, samples }
    }
    case 'assistant_message': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.sample_id)
      if (existing) {
        const messages = [...existing.messages, { turn: event.turn, content: event.content, timestamp: event.timestamp }]
        samples.set(event.sample_id, {
          ...existing,
          messages,
          sample: {
            ...existing.sample,
            trajectory: {
              ...existing.sample.trajectory,
              messages: [
                ...(existing.sample.trajectory.messages ?? []),
                {
                  role: 'assistant',
                  content: event.content,
                  timestamp: event.timestamp,
                },
              ],
            },
            metadata: {
              ...existing.sample.metadata,
              turns_used: event.turn,
              status: 'running',
            },
          },
        })
      }
      return { ...prev, samples }
    }
    case 'sample_end': {
      const samples = new Map(prev.samples)
      const existing = samples.get(event.id)
      if (existing) {
        samples.set(event.id, {
          ...existing,
          status: 'done',
          score: event.score,
          sample: {
            ...existing.sample,
            reward: event.score ?? undefined,
            status: 'completed',
            metadata: {
              ...existing.sample.metadata,
              status: 'completed',
              turns_used: existing.turn,
            },
          },
        })
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
