import { useState, useEffect } from 'react'
import { ChevronLeft } from 'lucide-react'
import { getRunReport } from '../api'
import type { RunReport } from '../types'

interface SampleRow {
  id: string
  reward: number | null
  turns: number | null
  tokens: number | null
  duration: number | null
  status: string | null
}

interface RunDetailProps {
  runId: string
  onBack: () => void
  onSelectSample: (sampleId: string) => void
}

function RewardCell({ reward }: { reward: number | null }) {
  if (reward === null) return <span style={{ color: 'var(--color-dark-text-muted)' }}>—</span>
  const pct = (reward * 100).toFixed(0)
  let color = 'var(--color-dark-text-muted)'
  if (reward >= 0.8) color = '#22c55e'
  else if (reward >= 0.5) color = '#3b82f6'
  return <span style={{ color, fontWeight: 500 }}>{pct}%</span>
}

export function RunDetail({ runId, onBack, onSelectSample }: RunDetailProps) {
  const [report, setReport] = useState<RunReport | null>(null)
  const [samples, setSamples] = useState<SampleRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    getRunReport(runId)
      .then(({ report: r, sample_ids }) => {
        setReport(r)
        // Build sample rows from report data
        // detailed per-sample stats come from the results JSONL via the /api/trace endpoint
        // For now seed rows from sample_ids; richer data fetched by RunViewer
        setSamples(
          sample_ids.map(id => ({ id, reward: null, turns: null, tokens: null, duration: null, status: null }))
        )
        setError(null)
      })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load run'))
      .finally(() => setLoading(false))
  }, [runId])

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
        <button onClick={onBack} className="text-sm mt-2" style={{ color: 'var(--color-dark-text-muted)' }}>← Back</button>
      </div>
    )
  }

  const metrics = report?.summary_metrics
  const model = report?.config?.endpoint?.model ?? '—'

  return (
    <div className="p-4 sm:p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={onBack}
          className="p-1.5 rounded hover:opacity-80 transition-opacity"
          style={{ color: 'var(--color-dark-text-muted)' }}
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div>
          <h1 className="text-xl font-semibold font-mono" style={{ color: 'var(--color-dark-text)' }}>
            {runId}
          </h1>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-dark-text-muted)' }}>
            {model}
          </p>
        </div>
      </div>

      {/* Summary metrics */}
      {metrics && (
        <div
          className="grid gap-px mb-6 rounded overflow-hidden"
          style={{
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            border: '1px solid var(--color-dark-border)',
            background: 'var(--color-dark-border)',
          }}
        >
          {[
            { label: 'Total Samples', value: metrics.total_samples },
            { label: 'Mean Reward', value: metrics.mean_reward !== undefined ? `${(metrics.mean_reward * 100).toFixed(1)}%` : '—' },
            { label: 'Avg Turns', value: metrics.avg_turns?.toFixed(1) ?? '—' },
            { label: 'Avg Tokens', value: metrics.avg_tokens !== undefined ? Math.round(metrics.avg_tokens).toLocaleString() : '—' },
            { label: 'Success Rate', value: metrics.success_rate !== undefined ? `${(metrics.success_rate * 100).toFixed(0)}%` : '—' },
            { label: 'Errors', value: metrics.provider_errors ?? 0 },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="px-3 py-3"
              style={{ background: 'var(--color-dark-card)' }}
            >
              <div className="text-[10px] uppercase tracking-wide mb-1" style={{ color: 'var(--color-dark-text-muted)' }}>
                {label}
              </div>
              <div className="text-base font-semibold font-mono" style={{ color: 'var(--color-dark-text)' }}>
                {String(value)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Samples table */}
      <div
        className="rounded overflow-hidden"
        style={{ border: '1px solid var(--color-dark-border)' }}
      >
        <div
          className="px-3 py-2 flex items-center justify-between"
          style={{ background: 'var(--color-dark-elevated)', borderBottom: '1px solid var(--color-dark-border)' }}
        >
          <span className="text-xs font-semibold" style={{ color: 'var(--color-dark-text)' }}>
            Samples ({samples.length})
          </span>
        </div>

        {samples.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--color-dark-text-muted)' }}>No samples found</p>
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead style={{ background: 'var(--color-dark-elevated)', borderBottom: '1px solid var(--color-dark-border)' }}>
              <tr>
                <th className="text-left py-2 px-3 text-xs font-semibold" style={{ color: 'var(--color-dark-text)' }}>
                  Sample ID
                </th>
                <th className="text-right py-2 px-3 text-xs font-semibold" style={{ color: 'var(--color-dark-text)' }}>
                  Reward
                </th>
                <th className="text-right py-2 px-3 text-xs font-semibold" style={{ color: 'var(--color-dark-text)' }}>
                  Turns
                </th>
                <th className="text-right py-2 px-3 text-xs font-semibold" style={{ color: 'var(--color-dark-text)' }}>
                  Tokens
                </th>
              </tr>
            </thead>
            <tbody style={{ background: 'var(--color-dark-card)' }}>
              {samples.map((s, i) => (
                <tr
                  key={s.id}
                  onClick={() => onSelectSample(s.id)}
                  className="cursor-pointer transition-colors"
                  style={{ borderTop: i > 0 ? '1px solid var(--color-dark-border)' : undefined }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-dark-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = '')}
                >
                  <td className="py-2 px-3">
                    <span className="font-mono text-xs" style={{ color: 'var(--color-dark-text)' }}>
                      {s.id}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right">
                    <RewardCell reward={s.reward} />
                  </td>
                  <td className="py-2 px-3 text-right text-xs" style={{ color: 'var(--color-dark-text-secondary)' }}>
                    {s.turns ?? '—'}
                  </td>
                  <td className="py-2 px-3 text-right text-xs" style={{ color: 'var(--color-dark-text-secondary)' }}>
                    {s.tokens !== null ? s.tokens.toLocaleString() : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
