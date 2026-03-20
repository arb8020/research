import { useEffect, useState } from 'react'
import { getRunReport } from '../api'
import type { RunReport, RunTags } from '../types'

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
      runId: string
      kind: 'loading'
      report: null
      samples: SampleRow[]
      tags: RunTags
      error: null
    }
  | {
      runId: string
      kind: 'loaded'
      report: RunReport
      samples: SampleRow[]
      tags: RunTags
      error: null
    }
  | {
      runId: string
      kind: 'error'
      report: null
      samples: SampleRow[]
      tags: RunTags
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
    runId,
    kind: 'loading',
    report: null,
    samples: [],
    tags: { user: {}, derived: {} },
    error: null,
  })

  useEffect(() => {
    let cancelled = false

    void getRunReport(runId)
      .then(({ report, sample_ids, tags }) => {
        if (cancelled) return
        setState({
          runId,
          kind: 'loaded',
          report,
          samples: buildSampleRows(sample_ids),
          tags,
          error: null,
        })
      })
      .catch(err => {
        if (cancelled) return
        setState({
          runId,
          kind: 'error',
          report: null,
          samples: [],
          tags: { user: {}, derived: {} },
          error: err instanceof Error ? err.message : 'Failed to load run',
        })
      })

    return () => {
      cancelled = true
    }
  }, [runId])

  if (state.runId !== runId) {
    return {
      runId,
      kind: 'loading',
      report: null,
      samples: [],
      tags: { user: {}, derived: {} },
      error: null,
    }
  }

  return state
}
