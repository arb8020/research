import { useEffect, useState } from 'react'
import { getSample, getWorkspace } from '../api'
import type { TraceSample, WorkspaceData } from '../types'

type SampleDataState =
  | {
      kind: 'loading'
      sample: null
      workspaceData: null
      error: null
    }
  | {
      kind: 'loaded'
      sample: TraceSample
      workspaceData: WorkspaceData | null
      error: null
    }
  | {
      kind: 'error'
      sample: null
      workspaceData: null
      error: string
    }

export function useSampleData(runId: string, sampleId: string): SampleDataState {
  const [state, setState] = useState<SampleDataState>({
    kind: 'loading',
    sample: null,
    workspaceData: null,
    error: null,
  })

  useEffect(() => {
    let cancelled = false

    setState({
      kind: 'loading',
      sample: null,
      workspaceData: null,
      error: null,
    })

    void Promise.all([
      getSample(runId, sampleId),
      getWorkspace(runId, sampleId).catch(() => null),
    ])
      .then(([sample, workspaceData]) => {
        if (cancelled) return
        setState({
          kind: 'loaded',
          sample,
          workspaceData,
          error: null,
        })
      })
      .catch(err => {
        if (cancelled) return
        setState({
          kind: 'error',
          sample: null,
          workspaceData: null,
          error: err instanceof Error ? err.message : 'Failed to load sample',
        })
      })

    return () => {
      cancelled = true
    }
  }, [runId, sampleId])

  return state
}
