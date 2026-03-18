import { useState, useEffect, useCallback } from 'react'
import { listRuns, listActiveRuns } from '../api'
import type { RunListItem, LiveRun } from '../types'

export function useRuns() {
  const [completedRuns, setCompletedRuns] = useState<RunListItem[]>([])
  const [liveRuns, setLiveRuns] = useState<LiveRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [completed, live] = await Promise.all([
        listRuns(),
        listActiveRuns(),
      ])
      setCompletedRuns(completed)
      setLiveRuns(live)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load runs')
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    void refresh()
  }, [refresh])

  // Poll for live run updates every 2s
  useEffect(() => {
    const interval = setInterval(() => {
      void listActiveRuns()
        .then(setLiveRuns)
        .catch(() => {})
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return { completedRuns, liveRuns, loading, error, refresh }
}
