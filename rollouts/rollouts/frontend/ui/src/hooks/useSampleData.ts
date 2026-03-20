import { useEffect, useState } from 'react'
import { getSample, getWorkspace } from '../api'
import type { TraceSample, WorkspaceData } from '../types'

type SampleDataState =
  | {
      runId: string
      sampleId: string
      kind: 'loading'
      sample: null
      workspaceData: null
      error: null
    }
  | {
      runId: string
      sampleId: string
      kind: 'loaded'
      sample: TraceSample
      workspaceData: WorkspaceData | null
      error: null
    }
  | {
      runId: string
      sampleId: string
      kind: 'error'
      sample: null
      workspaceData: null
      error: string
    }

export function useSampleData(runId: string, sampleId: string): SampleDataState {
  const [state, setState] = useState<SampleDataState>({
    runId,
    sampleId,
    kind: 'loading',
    sample: null,
    workspaceData: null,
    error: null,
  })

  useEffect(() => {
    let cancelled = false

    void Promise.all([
      getSample(runId, sampleId),
      getWorkspace(runId, sampleId).catch(() => null),
    ])
      .then(([sample, workspaceData]) => {
        if (cancelled) return
        setState({
          runId,
          sampleId,
          kind: 'loaded',
          sample,
          workspaceData,
          error: null,
        })
      })
      .catch(err => {
        if (cancelled) return
        setState({
          runId,
          sampleId,
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

  if (state.runId !== runId || state.sampleId !== sampleId) {
    return {
      runId,
      sampleId,
      kind: 'loading',
      sample: null,
      workspaceData: null,
      error: null,
    }
  }

  return state
}
