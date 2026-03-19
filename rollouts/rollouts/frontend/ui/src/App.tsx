import { useCallback, useEffect, useState } from 'react'
import { useRuns } from './hooks/useRuns'
import { RunsList } from './components/RunsList'
import { RunDetail } from './components/RunDetail'
import { RunViewer } from './components/RunViewer'
import { LiveSampleViewer } from './components/LiveSampleViewer'
import { ResultsDirPicker } from './components/ResultsDirPicker'
import { getResultsDirs, setResultsDir } from './api'

type View =
  | { kind: 'runs' }
  | { kind: 'run-detail'; runId: string }
  | { kind: 'sample'; runId: string; sampleId: string }
  | { kind: 'live-sample'; runId: string; sampleId: string }

type LocationState = {
  resultsDir: string | null
  view: View
}

function parseLocationState(search: string): LocationState {
  const params = new URLSearchParams(search)
  const resultsDir = params.get('results_dir')
  const runId = params.get('run_id')
  const sampleId = params.get('sample_id')

  if (runId && sampleId) {
    return { resultsDir, view: { kind: 'sample', runId, sampleId } }
  }
  if (runId) {
    return { resultsDir, view: { kind: 'run-detail', runId } }
  }
  return { resultsDir, view: { kind: 'runs' } }
}

function buildLocationSearch(resultsDir: string | null, view: View): string {
  const params = new URLSearchParams()
  if (resultsDir) params.set('results_dir', resultsDir)
  if (view.kind === 'run-detail' || view.kind === 'sample' || view.kind === 'live-sample') {
    params.set('run_id', view.runId)
  }
  if (view.kind === 'sample' || view.kind === 'live-sample') {
    params.set('sample_id', view.sampleId)
  }
  const query = params.toString()
  return query ? `?${query}` : ''
}

export default function App() {
  const [view, setView] = useState<View>({ kind: 'runs' })
  const [resultsDir, setCurrentResultsDir] = useState<string | null>(null)
  const { completedRuns, liveRuns, loading, error, refresh, autoPoll, setAutoPoll } = useRuns()

  const isSample = view.kind === 'sample' || view.kind === 'live-sample'
  const liveCount = liveRuns.filter(r => r.status === 'running' || r.status === 'watching').length

  const setUrlState = useCallback((nextResultsDir: string | null, nextView: View) => {
    const nextSearch = buildLocationSearch(nextResultsDir, nextView)
    const nextUrl = `${window.location.pathname}${nextSearch}`
    if (`${window.location.pathname}${window.location.search}` !== nextUrl) {
      window.history.replaceState(null, '', nextUrl)
    }
  }, [])

  const navigate = useCallback((nextView: View, nextResultsDir: string | null = resultsDir) => {
    setView(nextView)
    setUrlState(nextResultsDir, nextView)
  }, [resultsDir, setUrlState])

  const applyQueryParams = useCallback(async () => {
    const locationState = parseLocationState(window.location.search)

    if (locationState.resultsDir) {
      await setResultsDir(locationState.resultsDir)
      setCurrentResultsDir(locationState.resultsDir)
      await refresh()
    }

    setView(locationState.view)
  }, [refresh])

  useEffect(() => {
    let cancelled = false

    async function bootstrapFromQuery() {
      try {
        await applyQueryParams()
      } catch (err) {
        if (cancelled) return
        console.error('Failed to bootstrap view from query params', err)
      }
    }

    void bootstrapFromQuery()
    return () => {
      cancelled = true
    }
  }, [applyQueryParams])

  useEffect(() => {
    getResultsDirs()
      .then(data => {
        setCurrentResultsDir(current => current ?? data.current)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    const onPopState = () => {
      void applyQueryParams().catch(err => {
        console.error('Failed to sync view from browser history', err)
      })
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [applyQueryParams])

  return (
    <div
      className="flex flex-col"
      style={{
        height: '100vh',
        overflow: 'hidden',
        background: 'var(--color-dark-bg)',
        color: 'var(--color-dark-text)',
      }}
    >
      {/* Top nav — fixed height, never scrolls away */}
      <header
        className="flex-shrink-0 flex items-center px-4 sm:px-6 z-40"
        style={{
          height: 44,
          borderBottom: '1px solid var(--color-dark-border)',
          background: 'var(--color-dark-bg)',
        }}
      >
        <button
          onClick={() => { navigate({ kind: 'runs' }); void refresh() }}
          className="text-sm font-semibold tracking-tight hover:opacity-80 transition-opacity"
          style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-dark-text)' }}
        >
          rollouts
        </button>

        {/* Breadcrumb */}
        {view.kind !== 'runs' && (
          <div className="flex items-center gap-1 ml-3" style={{ color: 'var(--color-dark-text-muted)' }}>
            <span className="text-xs">/</span>
            <button
              onClick={() => {
                if (view.kind === 'sample') {
                  navigate({ kind: 'run-detail', runId: view.runId })
                } else {
                  navigate({ kind: 'runs' })
                }
              }}
              className="text-xs hover:opacity-80 transition-opacity font-mono truncate max-w-xs"
              style={{ color: 'var(--color-dark-text-secondary)' }}
            >
              {view.kind === 'run-detail' || view.kind === 'sample' || view.kind === 'live-sample' ? view.runId : ''}
            </button>
            {view.kind === 'sample' && (
              <>
                <span className="text-xs">/</span>
                <span className="text-xs font-mono" style={{ color: 'var(--color-dark-text-secondary)' }}>
                  {view.sampleId}
                </span>
              </>
            )}
          </div>
        )}

        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={() => { void refresh() }}
            className="text-xs px-2 py-1 rounded hover:opacity-80 transition-opacity"
            style={{
              color: 'var(--color-dark-text-secondary)',
              border: '1px solid var(--color-dark-border)',
              background: 'var(--color-dark-card)',
            }}
          >
            Sync
          </button>
          <button
            onClick={() => {
              const next = !autoPoll
              setAutoPoll(next)
              if (next) {
                void refresh()
              }
            }}
            className="text-xs px-2 py-1 rounded hover:opacity-80 transition-opacity"
            style={{
              color: autoPoll ? 'var(--color-dark-text)' : 'var(--color-dark-text-muted)',
              border: '1px solid var(--color-dark-border)',
              background: autoPoll ? 'var(--color-dark-elevated)' : 'var(--color-dark-card)',
            }}
          >
            Auto-poll {autoPoll ? 'On' : 'Off'}
          </button>
          <ResultsDirPicker onChanged={(path) => {
            setCurrentResultsDir(path)
            navigate({ kind: 'runs' }, path)
            void refresh()
          }} />
          {liveCount > 0 && (
            <div className="flex items-center gap-1.5">
              <div
                className="rounded-full"
                style={{
                  width: 7, height: 7,
                  background: '#22c55e',
                  animation: 'pulse-subtle 1.5s ease-in-out infinite',
                }}
              />
              <span className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
                {liveCount} live
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Page content — fills remaining height, scrollable for list/detail, locked for sample */}
      <div
        className="flex-1 min-h-0"
        style={{ overflowY: isSample ? 'hidden' : 'auto' }}
      >
        {view.kind === 'runs' && (
          <RunsList
            completedRuns={completedRuns}
            liveRuns={liveRuns}
            loading={loading}
            error={error}
            onSelectRun={runId => navigate({ kind: 'run-detail', runId })}
            onSelectLiveSample={(runId, sampleId) => navigate({ kind: 'live-sample', runId, sampleId })}
          />
        )}

        {view.kind === 'run-detail' && (
          <RunDetail
            runId={view.runId}
            onBack={() => navigate({ kind: 'runs' })}
            onSelectSample={sampleId => {
              if (view.kind === 'run-detail') {
                navigate({ kind: 'sample', runId: view.runId, sampleId })
              }
            }}
          />
        )}

        {view.kind === 'sample' && (
          <RunViewer
            runId={view.runId}
            sampleId={view.sampleId}
            evalName={completedRuns.find(r => r.id === view.runId)?.name}
            onBack={() => {
              if (view.kind === 'sample') {
                navigate({ kind: 'run-detail', runId: view.runId })
              }
            }}
          />
        )}

        {view.kind === 'live-sample' && (
          <LiveSampleViewer
            run={liveRuns.find(r => r.id === view.runId) ?? {
              id: view.runId,
              name: view.runId,
              timestamp: 0,
              total_samples: null,
              mean_reward: null,
              status: 'watching',
              live: true,
              can_kill: false,
            }}
            sampleId={view.sampleId}
            onBack={() => navigate({ kind: 'runs' })}
          />
        )}
      </div>
    </div>
  )
}
