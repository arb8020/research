import { useCallback, useEffect, useState } from 'react'
import { getResultsDirs } from '../api'
import type { ResultsDirsResponse } from '../api'

type ResultsDirsState =
  | {
      kind: 'loading'
      data: null
      error: null
    }
  | {
      kind: 'loaded'
      data: ResultsDirsResponse
      error: null
    }
  | {
      kind: 'error'
      data: null
      error: string
    }

export function useResultsDirs() {
  const [state, setState] = useState<ResultsDirsState>({
    kind: 'loading',
    data: null,
    error: null,
  })

  const refresh = useCallback(async () => {
    setState(prev => (prev.kind === 'loaded' ? prev : { kind: 'loading', data: null, error: null }))
    try {
      const data = await getResultsDirs()
      setState({ kind: 'loaded', data, error: null })
    } catch (err) {
      setState({
        kind: 'error',
        data: null,
        error: err instanceof Error ? err.message : 'Failed to load results directories',
      })
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { state, refresh }
}
