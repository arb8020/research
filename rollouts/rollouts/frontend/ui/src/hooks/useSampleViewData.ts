import { useSampleData } from './useSampleData'
import { useLiveRun } from './useLiveRun'
import type { RunListItem, TraceSample, WorkspaceData } from '../types'

type SampleViewState =
  | {
      runId: string
      sampleId: string
      kind: 'loading'
      sample: null
      workspaceData: null
      error: null
      source: 'live' | 'artifact' | null
      canExport: boolean
    }
  | {
      runId: string
      sampleId: string
      kind: 'loaded'
      sample: TraceSample
      workspaceData: WorkspaceData | null
      error: null
      source: 'live' | 'artifact'
      canExport: boolean
    }
  | {
      runId: string
      sampleId: string
      kind: 'error'
      sample: null
      workspaceData: null
      error: string
      source: 'live' | 'artifact' | null
      canExport: boolean
    }

export function useSampleViewData(
  runId: string,
  sampleId: string,
  liveStatus: RunListItem['status'] | null,
): SampleViewState {
  const liveState = useLiveRun(runId, liveStatus ?? 'completed')
  const liveSample = liveState.samples.get(sampleId)?.sample ?? null
  const liveRunActive = liveStatus === 'running' || liveStatus === 'watching'
  const liveStreamFinished =
    liveState.status === 'completed' ||
    liveState.status === 'failed' ||
    liveState.status === 'killed'

  const artifactState = useSampleData(runId, sampleId, {
    enabled: !liveRunActive || liveStreamFinished,
  })

  if (artifactState.kind === 'loaded') {
    return {
      ...artifactState,
      source: 'artifact',
      canExport: true,
    }
  }

  if (liveRunActive || liveSample !== null) {
    if (liveSample !== null) {
      return {
        runId,
        sampleId,
        kind: 'loaded',
        sample: liveSample,
        workspaceData: null,
        error: null,
        source: 'live',
        canExport: false,
      }
    }

    return {
      runId,
      sampleId,
      kind: 'loading',
      sample: null,
      workspaceData: null,
      error: null,
      source: 'live',
      canExport: false,
    }
  }

  if (artifactState.kind === 'error') {
    return {
      ...artifactState,
      source: null,
      canExport: false,
    }
  }

  return {
    ...artifactState,
    source: null,
    canExport: false,
  }
}
