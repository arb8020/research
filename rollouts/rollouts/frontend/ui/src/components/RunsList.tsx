import { useState, useMemo } from 'react'
import { LiveRunCard } from './LiveRunCard'
import type { RunListItem, LiveRun } from '../types'

interface RunsListProps {
  completedRuns: RunListItem[]
  liveRuns: LiveRun[]
  loading: boolean
  error: string | null
  onSelectRun: (runId: string) => void
  onSelectLiveSample?: (runId: string, sampleId: string) => void
}

type SortField = 'name' | 'timestamp' | 'samples' | 'reward'
type SortDir = 'asc' | 'desc'

function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function RewardBadge({ reward }: { reward: number }) {
  const pct = (reward * 100).toFixed(0)
  let color = 'var(--color-dark-text-muted)'
  if (reward >= 0.8) color = '#22c55e'
  else if (reward >= 0.5) color = '#3b82f6'
  return <span style={{ color, fontWeight: 500 }}>{pct}%</span>
}

export function RunsList({ completedRuns, liveRuns, loading, error, onSelectRun, onSelectLiveSample }: RunsListProps) {
  const [search, setSearch] = useState('')
  const [sortField, setSortField] = useState<SortField>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const filtered = useMemo(() => {
    let runs = completedRuns
    if (search.trim()) {
      const q = search.toLowerCase()
      runs = runs.filter(r => r.name.toLowerCase().includes(q))
    }
    return [...runs].sort((a, b) => {
      let av: string | number, bv: string | number
      switch (sortField) {
        case 'name': av = a.name; bv = b.name; break
        case 'timestamp': av = a.timestamp; bv = b.timestamp; break
        case 'samples': av = a.total_samples; bv = b.total_samples; break
        case 'reward': av = a.mean_reward; bv = b.mean_reward; break
        default: return 0
      }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
  }, [completedRuns, search, sortField, sortDir])

  function SortIndicator({ field }: { field: SortField }) {
    if (sortField !== field) return <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 9 }}>⇅</span>
    return <span style={{ color: 'var(--color-dark-text-secondary)', fontSize: 9 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ height: 200 }}>
        <div
          className="rounded-full animate-spin"
          style={{ width: 24, height: 24, border: '2px solid var(--color-dark-border)', borderTopColor: 'var(--color-dark-text)' }}
        />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: '#ef4444' }}>Error: {error}</p>
      </div>
    )
  }

  return (
    <div className="p-4 sm:p-6 max-w-5xl mx-auto">
      {/* Live runs section */}
      {liveRuns.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div
              className="rounded-full"
              style={{
                width: 8, height: 8,
                background: '#22c55e',
                animation: 'pulse-subtle 1.5s ease-in-out infinite',
              }}
            />
            <h2 className="text-sm font-semibold" style={{ color: 'var(--color-dark-text)' }}>
              Live ({liveRuns.length})
            </h2>
          </div>
          <div className="space-y-2">
            {liveRuns.map(run => (
              <LiveRunCard key={run.run_id} run={run} onSelectSample={onSelectLiveSample ? (sampleId) => onSelectLiveSample(run.run_id, sampleId) : undefined} />
            ))}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-semibold" style={{ color: 'var(--color-dark-text)' }}>
            Runs
          </h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--color-dark-text-muted)' }}>
            {completedRuns.length} completed evaluation{completedRuns.length !== 1 ? 's' : ''}
          </p>
        </div>
        <input
          type="text"
          placeholder="Search runs…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="text-sm px-3 py-1.5 rounded"
          style={{
            background: 'var(--color-dark-card)',
            border: '1px solid var(--color-dark-border)',
            color: 'var(--color-dark-text)',
            outline: 'none',
            width: 220,
          }}
        />
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-16">
          <p className="text-sm" style={{ color: 'var(--color-dark-text-muted)' }}>
            {search ? 'No runs match your search' : 'No completed runs yet'}
          </p>
        </div>
      ) : (
        <div
          className="overflow-x-auto rounded"
          style={{ border: '1px solid var(--color-dark-border)' }}
        >
          <table className="w-full text-sm border-collapse" style={{ minWidth: 600 }}>
            <thead style={{ background: 'var(--color-dark-elevated)', borderBottom: '1px solid var(--color-dark-border)' }}>
              <tr>
                <th
                  className="text-left py-2 px-3 cursor-pointer text-xs font-semibold select-none"
                  style={{ color: 'var(--color-dark-text)' }}
                  onClick={() => handleSort('name')}
                >
                  <span className="flex items-center gap-1">
                    Run <SortIndicator field="name" />
                  </span>
                </th>
                <th
                  className="text-left py-2 px-3 cursor-pointer text-xs font-semibold select-none"
                  style={{ color: 'var(--color-dark-text)' }}
                  onClick={() => handleSort('timestamp')}
                >
                  <span className="flex items-center gap-1">
                    Time <SortIndicator field="timestamp" />
                  </span>
                </th>
                <th
                  className="text-right py-2 px-3 cursor-pointer text-xs font-semibold select-none"
                  style={{ color: 'var(--color-dark-text)' }}
                  onClick={() => handleSort('samples')}
                >
                  <span className="flex items-center justify-end gap-1">
                    Samples <SortIndicator field="samples" />
                  </span>
                </th>
                <th
                  className="text-right py-2 px-3 cursor-pointer text-xs font-semibold select-none"
                  style={{ color: 'var(--color-dark-text)' }}
                  onClick={() => handleSort('reward')}
                >
                  <span className="flex items-center justify-end gap-1">
                    Reward <SortIndicator field="reward" />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody style={{ background: 'var(--color-dark-card)' }}>
              {filtered.map((run, i) => (
                <tr
                  key={run.id}
                  onClick={() => onSelectRun(run.id)}
                  className="cursor-pointer transition-colors"
                  style={{
                    borderTop: i > 0 ? '1px solid var(--color-dark-border)' : undefined,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-dark-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = '')}
                >
                  <td className="py-2 px-3">
                    <span className="font-mono text-xs" style={{ color: 'var(--color-dark-text)' }}>
                      {run.name}
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    <span className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
                      {formatTimestamp(run.timestamp)}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right">
                    <span className="text-sm" style={{ color: 'var(--color-dark-text)' }}>
                      {run.total_samples}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right">
                    <RewardBadge reward={run.mean_reward} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Sort indicator legend */}
      <div className="flex items-center gap-1 mt-2">
        {sortField !== 'timestamp' && (
          <button
            onClick={() => { setSortField('timestamp'); setSortDir('desc') }}
            className="text-[10px] hover:opacity-80"
            style={{ color: 'var(--color-dark-text-muted)' }}
          >
            Reset sort
          </button>
        )}
      </div>
    </div>
  )
}
