import { useState } from 'react'
import { useRuns } from './hooks/useRuns'
import { RunsList } from './components/RunsList'
import { RunDetail } from './components/RunDetail'
import { RunViewer } from './components/RunViewer'
import { ResultsDirPicker } from './components/ResultsDirPicker'

type View =
  | { kind: 'runs' }
  | { kind: 'run-detail'; runId: string }
  | { kind: 'sample'; runId: string; sampleId: string }

export default function App() {
  const [view, setView] = useState<View>({ kind: 'runs' })
  const { completedRuns, liveRuns, loading, error, refresh } = useRuns()

  const isSample = view.kind === 'sample'

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
          onClick={() => { setView({ kind: 'runs' }); void refresh() }}
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
                  setView({ kind: 'run-detail', runId: view.runId })
                } else {
                  setView({ kind: 'runs' })
                }
              }}
              className="text-xs hover:opacity-80 transition-opacity font-mono truncate max-w-xs"
              style={{ color: 'var(--color-dark-text-secondary)' }}
            >
              {view.kind === 'run-detail' || view.kind === 'sample' ? view.runId : ''}
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
          <ResultsDirPicker onChanged={() => { setView({ kind: 'runs' }); void refresh() }} />
          {liveRuns.filter(r => r.status === 'running').length > 0 && (
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
                {liveRuns.filter(r => r.status === 'running').length} running
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
            onSelectRun={runId => setView({ kind: 'run-detail', runId })}
          />
        )}

        {view.kind === 'run-detail' && (
          <RunDetail
            runId={view.runId}
            onBack={() => setView({ kind: 'runs' })}
            onSelectSample={sampleId => {
              if (view.kind === 'run-detail') {
                setView({ kind: 'sample', runId: view.runId, sampleId })
              }
            }}
          />
        )}

        {view.kind === 'sample' && (
          <RunViewer
            runId={view.runId}
            sampleId={view.sampleId}
            onBack={() => {
              if (view.kind === 'sample') {
                setView({ kind: 'run-detail', runId: view.runId })
              }
            }}
          />
        )}
      </div>
    </div>
  )
}
