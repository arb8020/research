import { useState } from 'react'
import { ChevronDown, ChevronUp, Square } from 'lucide-react'
import { useLiveRun } from '../hooks/useLiveRun'
import { killRun } from '../api'
import type { LiveRun } from '../types'

interface LiveRunCardProps {
  run: LiveRun
  onFinished?: () => void
}

function elapsed(startTime: number): string {
  const secs = Math.floor(Date.now() / 1000 - startTime)
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function ScoreDot({ score, status }: { score: number | null; status: 'pending' | 'running' | 'done' }) {
  let bg = 'var(--color-dark-elevated)'
  let title = 'pending'

  if (status === 'running') {
    bg = 'var(--color-neutral-500)'
    title = 'running'
  } else if (status === 'done' && score !== null) {
    if (score >= 0.8) { bg = '#22c55e'; title = `score: ${score.toFixed(2)}` }
    else if (score >= 0.4) { bg = '#3b82f6'; title = `score: ${score.toFixed(2)}` }
    else { bg = '#ef4444'; title = `score: ${score.toFixed(2)}` }
  }

  return (
    <div
      className="rounded-sm"
      style={{ width: 10, height: 10, background: bg, flexShrink: 0 }}
      title={title}
    />
  )
}

export function LiveRunCard({ run, onFinished: _onFinished }: LiveRunCardProps) {
  const state = useLiveRun(run.run_id, run.status)
  const [expanded, setExpanded] = useState(true)
  const [killing, setKilling] = useState(false)

  const samples = Array.from(state.samples.values())
  const done = samples.filter(s => s.status === 'done').length
  const total = state.total ?? run.output_length ?? samples.length
  const progress = total > 0 ? done / total : 0

  const statusColor = {
    running: '#22c55e',
    watching: '#3b82f6',
    completed: 'var(--color-dark-text-muted)',
    failed: '#ef4444',
    killed: 'var(--color-neutral-500)',
  }[state.status] ?? 'var(--color-dark-text-muted)'

  const handleKill = async () => {
    setKilling(true)
    try {
      await killRun(run.run_id)
    } catch {
      // ignore
    } finally {
      setKilling(false)
    }
  }

  // Scores of finished samples for mean
  const scores = samples.filter(s => s.score !== null).map(s => s.score as number)
  const meanScore = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : null

  return (
    <div
      className="rounded"
      style={{
        border: '1px solid var(--color-dark-border)',
        background: 'var(--color-dark-card)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-3 py-2 cursor-pointer"
        onClick={() => setExpanded(e => !e)}
      >
        {/* Pulse dot */}
        <div className="flex-shrink-0 relative">
          <div
            className="rounded-full"
            style={{
              width: 8,
              height: 8,
              background: statusColor,
              ...(state.status === 'running' || run.status === 'watching' ? { animation: 'pulse-subtle 1.5s ease-in-out infinite' } : {}),
            }}
          />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className="text-sm font-medium truncate"
              style={{ color: 'var(--color-dark-text)' }}
            >
              {run.config_name}
            </span>
            <span
              className="text-xs flex-shrink-0"
              style={{ color: 'var(--color-dark-text-muted)' }}
            >
              {elapsed(run.start_time)}
            </span>
            {state.status !== 'running' && run.status !== 'watching' && (
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  background: 'var(--color-dark-elevated)',
                  color: statusColor,
                  fontSize: '10px',
                }}
              >
                {state.status}
              </span>
            )}
          </div>

          {/* Progress bar */}
          {total > 0 && (
            <div className="flex items-center gap-2 mt-1">
              <div
                className="flex-1 rounded"
                style={{ height: 3, background: 'var(--color-dark-elevated)' }}
              >
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{ width: `${progress * 100}%`, background: statusColor }}
                />
              </div>
              <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--color-dark-text-muted)' }}>
                {done}/{total}
                {meanScore !== null && ` · avg ${meanScore.toFixed(2)}`}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          {state.status === 'running' && (
            <button
              onClick={e => { e.stopPropagation(); void handleKill() }}
              disabled={killing}
              className="p-1 rounded hover:opacity-80 transition-opacity"
              style={{ color: 'var(--color-dark-text-muted)' }}
              title="Kill run"
            >
              <Square className="h-3 w-3" />
            </button>
          )}
          {expanded
            ? <ChevronUp className="h-4 w-4" style={{ color: 'var(--color-dark-text-muted)' }} />
            : <ChevronDown className="h-4 w-4" style={{ color: 'var(--color-dark-text-muted)' }} />
          }
        </div>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div
          className="px-3 pb-3"
          style={{ borderTop: '1px solid var(--color-dark-border)' }}
        >
          {/* Sample status grid */}
          {samples.length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] mb-1.5" style={{ color: 'var(--color-dark-text-muted)' }}>
                Samples
              </div>
              <div className="flex flex-wrap gap-1">
                {samples.map(s => (
                  <ScoreDot key={s.id} score={s.score} status={s.status} />
                ))}
              </div>
              {/* Score breakdown */}
              {scores.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {samples.filter(s => s.status === 'done').map(s => (
                    <span
                      key={s.id}
                      className="text-[10px] font-mono"
                      style={{ color: (s.score ?? 0) >= 0.8 ? '#22c55e' : (s.score ?? 0) >= 0.4 ? '#3b82f6' : '#ef4444' }}
                    >
                      {s.id.replace('sample_', '')}: {s.score?.toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Recent stdout */}
          {state.stdout_lines.length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] mb-1" style={{ color: 'var(--color-dark-text-muted)' }}>
                Output
              </div>
              <div
                className="font-mono text-[10px] overflow-y-auto"
                style={{
                  maxHeight: 120,
                  color: 'var(--color-dark-text-secondary)',
                  background: 'var(--color-dark-elevated)',
                  padding: '6px 8px',
                  borderRadius: '2px',
                }}
              >
                {state.stdout_lines.slice(-20).map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
