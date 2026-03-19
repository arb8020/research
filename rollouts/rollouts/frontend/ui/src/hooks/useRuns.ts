import { useState, useEffect, useCallback } from 'react'
import { listRuns, listActiveRuns } from '../api'
import type { RunListItem, LiveRun } from '../types'

type RunsState =
  | {
      kind: 'loading'
      completedRuns: RunListItem[]
      liveRuns: LiveRun[]
      error: null
    }
  | {
      kind: 'loaded'
      completedRuns: RunListItem[]
      liveRuns: LiveRun[]
      error: null
    }
  | {
      kind: 'error'
      completedRuns: RunListItem[]
      liveRuns: LiveRun[]
      error: string
    }

export function useRuns() {
  const [state, setState] = useState<RunsState>({
    kind: 'loading',
    completedRuns: [],
    liveRuns: [],
    error: null,
  })
  const [autoPoll, setAutoPoll] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [completed, live] = await Promise.all([
        listRuns(),
        listActiveRuns(),
      ])
      setState({
        kind: 'loaded',
        completedRuns: completed,
        liveRuns: live,
        error: null,
      })
    } catch (err) {
      setState(prev => ({
        kind: 'error',
        completedRuns: prev.completedRuns,
        liveRuns: prev.liveRuns,
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
      void listActiveRuns()
        .then(liveRuns => {
          setState(prev => ({ ...prev, liveRuns }))
        })
        .catch(err => {
          console.error('Failed to refresh live runs', err)
        })
    }, 2000)
    return () => clearInterval(interval)
  }, [autoPoll])

  return {
    completedRuns: state.completedRuns,
    liveRuns: state.liveRuns,
    loading: state.kind === 'loading',
    error: state.error,
    autoPoll,
    setAutoPoll,
    refresh,
  }
}
