import { useState, useEffect, useCallback } from 'react'
import { listRuns } from '../api'
import type { RunListItem } from '../types'

type RunsState =
  | {
      kind: 'loading'
      runs: RunListItem[]
      error: null
    }
  | {
      kind: 'loaded'
      runs: RunListItem[]
      error: null
    }
  | {
      kind: 'error'
      runs: RunListItem[]
      error: string
    }

export function useRuns() {
  const [state, setState] = useState<RunsState>({
    kind: 'loading',
    runs: [],
    error: null,
  })
  const [autoPoll, setAutoPoll] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const runs = await listRuns()
      setState({
        kind: 'loaded',
        runs,
        error: null,
      })
    } catch (err) {
      setState(prev => ({
        kind: 'error',
        runs: prev.runs,
        error: err instanceof Error ? err.message : 'Failed to load runs',
      }))
    }
  }, [])

  // Initial load
  useEffect(() => {
    void refresh()
  }, [refresh])

  // Poll for live run updates every 2s when explicitly enabled
  useEffect(() => {
    if (!autoPoll) return

    const interval = setInterval(() => {
      void listRuns()
        .then(runs => {
          setState(prev => ({ ...prev, runs }))
        })
        .catch(err => {
          console.error('Failed to refresh runs', err)
        })
    }, 2000)
    return () => clearInterval(interval)
  }, [autoPoll])

  const completedRuns = state.runs.filter(run => !run.live)
  const liveRuns = state.runs.filter(run => run.live)

  return {
    completedRuns,
    liveRuns,
    runs: state.runs,
    loading: state.kind === 'loading',
    error: state.error,
    autoPoll,
    setAutoPoll,
    refresh,
  }
}
