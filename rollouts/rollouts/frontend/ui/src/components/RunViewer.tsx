import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react'
import { ChevronLeft } from 'lucide-react'
import { getSample, getWorkspace } from '../api'
import { ConversationView } from './ConversationView'
import { WorkspacePanel } from './WorkspacePanel'
import { ErrorBoundary } from './ErrorBoundary'
import type { TraceSample, WorkspaceData } from '../types'

const SPLIT_KEY = 'rollouts-split-pct'
const DEFAULT_SPLIT = 54  // left panel % width

function useSplitPct(): [number, (pct: number) => void] {
  const [pct, setPct] = useState(() => {
    try { return Number(localStorage.getItem(SPLIT_KEY)) || DEFAULT_SPLIT } catch { return DEFAULT_SPLIT }
  })
  const set = useCallback((v: number) => {
    setPct(v)
    try { localStorage.setItem(SPLIT_KEY, String(v)) } catch {}
  }, [])
  return [pct, set]
}

function SummaryField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide mb-1 font-medium" style={{ color: 'var(--color-dark-text-muted)' }}>
        {label}
      </div>
      {children}
    </div>
  )
}

interface RunViewerProps {
  runId: string
  sampleId: string
  onBack: () => void
}

type TabId = 'conversation' | 'summary' | 'json'

export function RunViewer({ runId, sampleId, onBack }: RunViewerProps) {
  const [activeTab, setActiveTab] = useState<TabId>('conversation')
  const [sample, setSample] = useState<TraceSample | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [workspaceData, setWorkspaceData] = useState<WorkspaceData | null>(null)
  const [selectedTurn, setSelectedTurn] = useState(0)
  const [conversationRef, setConversationRef] = useState<{ scrollToMessage: (idx: number) => void } | null>(null)
  const [splitPct, setSplitPct] = useSplitPct()
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setLoading(true)
    getSample(runId, sampleId)
      .then(s => { setSample(s); setError(null) })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load sample'))
      .finally(() => setLoading(false))
    // Load workspace data in parallel (non-blocking — not all runs have workspace)
    getWorkspace(runId, sampleId)
      .then(w => setWorkspaceData(w))
      .catch(() => setWorkspaceData(null))
  }, [runId, sampleId])

  const handleJumpToMessage = useCallback((messageIndex: number) => {
    conversationRef?.scrollToMessage(messageIndex)
    setActiveTab('conversation')
  }, [conversationRef])

  const onDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    const onMove = (ev: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const pct = Math.min(80, Math.max(20, ((ev.clientX - rect.left) / rect.width) * 100))
      setSplitPct(Math.round(pct))
    }
    const onUp = () => { dragging.current = false; window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [setSplitPct])

  const tabs: { id: TabId; label: string }[] = [
    { id: 'conversation', label: 'Conversation' },
    { id: 'summary', label: 'Summary' },
    { id: 'json', label: 'JSON' },
  ]

  // Track which conversation message index maps to which turn
  const messageToTurn = useCallback((messageIndex: number) => {
    // Each assistant message = one turn. Count assistant messages up to messageIndex.
    const messages = sample?.trajectory?.messages ?? []
    let turn = 0
    for (let i = 0; i <= messageIndex && i < messages.length; i++) {
      if (messages[i]?.role === 'assistant') turn++
    }
    return Math.max(0, turn - 1)
  }, [sample])

  const hasWorkspace = workspaceData !== null && workspaceData.snapshots.length > 0

  // Header + tabs shared by both layouts
  const header = (
    <>
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={onBack}
          className="p-1.5 rounded hover:opacity-80 transition-opacity"
          style={{ color: 'var(--color-dark-text-muted)' }}
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div>
          <h1 className="text-lg font-semibold font-mono" style={{ color: 'var(--color-dark-text)' }}>
            {sampleId}
          </h1>
          <p className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
            {runId}
          </p>
        </div>
      </div>
      <div className="flex gap-1 mb-4" style={{ borderBottom: '1px solid var(--color-dark-border)' }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="px-3 py-2 text-sm font-medium transition-colors"
            style={{
              borderBottom: activeTab === tab.id ? '2px solid var(--color-dark-text)' : '2px solid transparent',
              color: activeTab === tab.id ? 'var(--color-dark-text)' : 'var(--color-dark-text-muted)',
              marginBottom: -1,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </>
  )

  // Non-workspace layout (simple, scrollable)
  if (!hasWorkspace) {
    return (
      <div className="p-4 sm:p-6 max-w-5xl mx-auto">
        {header}
        <div className="rounded p-4" style={{ background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>
          {loading ? (
            <div className="flex items-center justify-center" style={{ height: 160 }}>
              <div className="rounded-full animate-spin" style={{ width: 24, height: 24, border: '2px solid var(--color-dark-border)', borderTopColor: 'var(--color-dark-text)' }} />
            </div>
          ) : error ? (
            <p className="text-sm" style={{ color: '#ef4444' }}>{error}</p>
          ) : sample ? (
            <>
              {activeTab === 'conversation' && (
                <ErrorBoundary label="ConversationView">
                  <ConversationView sample={sample} />
                </ErrorBoundary>
              )}
              {activeTab === 'summary' && (
                <ErrorBoundary label="Summary">
                  <div className="space-y-4">
                    {sample.reward != null && (
                      <SummaryField label="Reward">
                        <span className="font-mono font-semibold text-sm" style={{ color: sample.reward >= 0.8 ? '#22c55e' : sample.reward >= 0.4 ? '#3b82f6' : 'var(--color-dark-text)' }}>
                          {(sample.reward as number).toFixed(3)}
                        </span>
                      </SummaryField>
                    )}
                    {sample.trajectory.completions.length > 0 && (
                      <SummaryField label="Completions">
                        <p className="text-sm font-mono" style={{ color: 'var(--color-dark-text)' }}>{sample.trajectory.completions.length}</p>
                      </SummaryField>
                    )}
                  </div>
                </ErrorBoundary>
              )}
              {activeTab === 'json' && (
                <div className="overflow-auto font-mono text-xs" style={{ background: '#0a0a0a', padding: '0.75rem', borderRadius: '2px', color: 'var(--color-neutral-300)', maxHeight: '70vh' }}>
                  <pre>{JSON.stringify(sample, null, 2)}</pre>
                </div>
              )}
            </>
          ) : null}
        </div>
      </div>
    )
  }

  // Workspace layout — full height, two locked panels
  return (
    <div className="flex flex-col" style={{ height: '100%', overflow: 'hidden' }}>
      {/* Header bar — never scrolls */}
      <div className="flex-shrink-0 px-4 pt-3">
        {header}
      </div>

      {/* Split content — fills remaining height */}
      <div className="flex-1 min-h-0 px-4 pb-3">
      {loading ? (
        <div className="flex items-center justify-center rounded" style={{ height: 160, background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>
          <div className="rounded-full animate-spin" style={{ width: 24, height: 24, border: '2px solid var(--color-dark-border)', borderTopColor: 'var(--color-dark-text)' }} />
        </div>
      ) : error ? (
        <p className="text-sm rounded p-4" style={{ color: '#ef4444', background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>{error}</p>
      ) : sample ? (
        <div
          ref={containerRef}
          className="flex"
          style={{ height: '100%', alignItems: 'stretch' }}
        >
          {/* Left panel — conversation/summary/json */}
          <div
            className="rounded p-4"
            style={{
              background: 'var(--color-dark-card)',
              border: '1px solid var(--color-dark-border)',
              width: `${splitPct}%`,
              flexShrink: 0,
              minWidth: 0,
              overflowY: 'auto',
              height: '100%',
            }}
          >
          <>
            {activeTab === 'conversation' && (
              <ErrorBoundary label="ConversationView">
                <ConversationView
                  sample={sample}
                  onMessageVisible={hasWorkspace ? (idx) => setSelectedTurn(messageToTurn(idx)) : undefined}
                />
              </ErrorBoundary>
            )}

            {activeTab === 'summary' && (
              <ErrorBoundary label="Summary">
              <div className="space-y-4">
                {/* Reward — scalar, present in both rollouts and charisma */}
                {sample.reward != null && (
                  <SummaryField label="Reward">
                    <span
                      className="font-mono font-semibold text-sm"
                      style={{ color: sample.reward >= 0.8 ? '#22c55e' : sample.reward >= 0.4 ? '#3b82f6' : 'var(--color-dark-text)' }}
                    >
                      {(sample.reward as number).toFixed(3)}
                    </span>
                  </SummaryField>
                )}

                {/* Score metrics (charisma: score.metrics; rollouts: rewards[]) */}
                {(() => {
                  const metrics = sample.score?.metrics ?? sample.rewards ?? []
                  if (metrics.length === 0) return null
                  return (
                    <SummaryField label="Score breakdown">
                      <div className="space-y-1">
                        {metrics.map((r, i) => (
                          <div key={i} className="flex items-center gap-3 text-sm">
                            <span style={{ color: 'var(--color-dark-text-secondary)' }}>{r.name}</span>
                            <span
                              className="font-mono font-semibold"
                              style={{ color: r.value >= 0.8 ? '#22c55e' : r.value >= 0.4 ? '#3b82f6' : 'var(--color-dark-text)' }}
                            >
                              {r.value.toFixed(3)}
                            </span>
                            {r.weight !== 1.0 && (
                              <span className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>w={r.weight}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </SummaryField>
                  )
                })()}

                {/* Input fields (rollouts) — skip if null */}
                {sample.input != null && Object.keys(sample.input).length > 0 && (
                  <SummaryField label="Input">
                    <div className="space-y-2">
                      {Object.entries(sample.input).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--color-dark-text-muted)' }}>{k}: </span>
                          <span className="text-sm" style={{ color: 'var(--color-dark-text)' }}>
                            {typeof v === 'string' ? v : JSON.stringify(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </SummaryField>
                )}

                {/* Metadata */}
                {sample.metadata != null && Object.keys(sample.metadata).length > 0 && (
                  <SummaryField label="Metadata">
                    <div className="space-y-1">
                      {Object.entries(sample.metadata).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2 text-sm flex-wrap">
                          <span style={{ color: 'var(--color-dark-text-secondary)' }}>{k}</span>
                          <span className="font-mono" style={{ color: 'var(--color-dark-text)' }}>{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  </SummaryField>
                )}

                {/* Completions count */}
                <SummaryField label="Completions">
                  <p className="text-sm font-mono" style={{ color: 'var(--color-dark-text)' }}>
                    {sample.trajectory.completions.length}
                  </p>
                </SummaryField>
              </div>
              </ErrorBoundary>
            )}

            {activeTab === 'json' && (
              <div
                className="overflow-auto font-mono text-xs"
                style={{
                  background: '#0a0a0a',
                  padding: '0.75rem',
                  borderRadius: '2px',
                  color: 'var(--color-neutral-300)',
                  maxHeight: '70vh',
                }}
              >
                <pre>{JSON.stringify(sample, null, 2)}</pre>
              </div>
            )}
          </>
          </div>

          {/* Drag divider */}
          {hasWorkspace && (
            <div
              onMouseDown={onDividerMouseDown}
              style={{
                width: 5,
                flexShrink: 0,
                cursor: 'col-resize',
                background: 'transparent',
                transition: 'background 100ms',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-dark-border)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />
          )}

          {/* Workspace panel (right, only when workspace data exists) */}
          {hasWorkspace && (
            <div style={{ flex: 1, minWidth: 0, height: '100%' }}>
              <ErrorBoundary label="WorkspacePanel">
                <WorkspacePanel
                  workspaceData={workspaceData!}
                  selectedTurn={selectedTurn}
                  onJumpToMessage={handleJumpToMessage}
                />
              </ErrorBoundary>
            </div>
          )}
        </div>
      ) : null}
      </div>{/* end content wrapper */}
    </div>
  )
}
