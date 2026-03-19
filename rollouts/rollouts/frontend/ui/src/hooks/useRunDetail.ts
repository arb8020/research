import { useEffect, useState } from 'react'
import { getRunReport } from '../api'
import type { RunReport } from '../types'

export interface SampleRow {
  id: string
  reward: number | null
  turns: number | null
  tokens: number | null
  duration: number | null
  status: string | null
}

type RunDetailState =
  | {
      kind: 'loading'
      report: null
      samples: SampleRow[]
      error: null
    }
  | {
      kind: 'loaded'
      report: RunReport
      samples: SampleRow[]
      error: null
    }
  | {
      kind: 'error'
      report: null
      samples: SampleRow[]
      error: string
    }

function buildSampleRows(sampleIds: string[]): SampleRow[] {
  return sampleIds.map(id => ({
    id,
    reward: null,
    turns: null,
    tokens: null,
    duration: null,
    status: null,
  }))
}

export function useRunDetail(runId: string): RunDetailState {
  const [state, setState] = useState<RunDetailState>({
    kind: 'loading',
    report: null,
    samples: [],
    error: null,
  })

  useEffect(() => {
    let cancelled = false

    setState({
      kind: 'loading',
      report: null,
      samples: [],
      error: null,
    })

    void getRunReport(runId)
      .then(({ report, sample_ids }) => {
        if (cancelled) return
        setState({
          kind: 'loaded',
          report,
          samples: buildSampleRows(sample_ids),
          error: null,
        })
      })
      .catch(err => {
        if (cancelled) return
        setState({
          kind: 'error',
          report: null,
          samples: [],
          error: err instanceof Error ? err.message : 'Failed to load run',
        })
      })

    return () => {
      cancelled = true
    }
  }, [runId])

  return state
}
