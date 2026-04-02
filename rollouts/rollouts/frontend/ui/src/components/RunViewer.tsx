import { useState, useCallback, useMemo, useRef, type ReactNode } from 'react'
import { ChevronLeft, Download } from 'lucide-react'
import { ConversationView } from './ConversationView'
import { WorkspacePanel } from './WorkspacePanel'
import { ErrorBoundary } from './ErrorBoundary'
import { resolvePlugin } from '../plugins/registry'
import { getSampleHtmlExportUrl } from '../api'
import { useSampleViewData } from '../hooks/useSampleViewData'
import type { RunListItem, TraceSample } from '../types'

const SPLIT_KEY = 'rollouts-split-pct'
const DEFAULT_SPLIT = 54  // left panel % width

function useSplitPct(): [number, (pct: number) => void] {
  const [pct, setPct] = useState(() => {
    try { return Number(localStorage.getItem(SPLIT_KEY)) || DEFAULT_SPLIT } catch { return DEFAULT_SPLIT }
  })
  const set = useCallback((v: number) => {
    setPct(v)
    try {
      localStorage.setItem(SPLIT_KEY, String(v))
    } catch {
      return
    }
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
  evalName?: string
  liveStatus?: RunListItem['status'] | null
  onBack: () => void
}

type TabId = 'conversation' | 'summary' | 'json'

function hasEnvironmentState(sample: TraceSample): boolean {
  return sample.environment_state != null
}

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }

function getVerifierTurnHistory(sample: TraceSample): Record<string, JsonValue>[] {
  const environmentState = sample.environment_state as Record<string, unknown> | null | undefined
  const verifierState = environmentState?.['verifier_state']
  if (!verifierState || typeof verifierState !== 'object') return []
  const turnHistory = (verifierState as Record<string, unknown>).turn_history
  return Array.isArray(turnHistory)
    ? turnHistory.filter((entry): entry is Record<string, JsonValue> => entry != null && typeof entry === 'object')
    : []
}

function getTurnEntry(turnHistory: Record<string, JsonValue>[], uiTurn: number): Record<string, JsonValue> | null {
  const targetTurn = uiTurn + 1
  return turnHistory.find(entry => entry.turn === targetTurn) ?? null
}

function JsonSection({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <div
        className="text-[10px] uppercase tracking-wide mb-2 font-medium"
        style={{ color: 'var(--color-dark-text-muted)' }}
      >
        {label}
      </div>
      <div
        className="overflow-auto font-mono text-xs rounded"
        style={{
          padding: '0.75rem',
          background: '#0a0a0a',
          border: '1px solid var(--color-dark-border)',
          maxHeight: '32vh',
        }}
      >
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </div>
    </div>
  )
}

function EnvironmentStatePanel({
  sample,
  selectedTurn,
  checkedTurns,
}: {
  sample: TraceSample
  selectedTurn: number
  checkedTurns: number[]
}) {
  const turnHistory = getVerifierTurnHistory(sample)
  const fullState = sample.environment_state

  const content = (() => {
    if (turnHistory.length === 0) {
      return <JsonSection label="Raw environment state" value={fullState} />
    }

    if (checkedTurns.length >= 2) {
      const startTurn = checkedTurns[0]
      const endTurn = checkedTurns[checkedTurns.length - 1]
      const startEntry = getTurnEntry(turnHistory, startTurn)
      const endEntry = getTurnEntry(turnHistory, endTurn)
      return (
        <>
          <JsonSection label={`Turn ${startTurn + 1} state`} value={startEntry ?? { turn: startTurn + 1, missing: true }} />
          <JsonSection label={`Turn ${endTurn + 1} state`} value={endEntry ?? { turn: endTurn + 1, missing: true }} />
        </>
      )
    }

    const focusTurn = checkedTurns.length === 1 ? checkedTurns[0] : selectedTurn
    const focusEntry = getTurnEntry(turnHistory, focusTurn)
    return <JsonSection label={`Turn ${focusTurn + 1} state`} value={focusEntry ?? { turn: focusTurn + 1, missing: true }} />
  })()

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div
        className="flex-shrink-0 px-3 py-2 text-xs font-medium uppercase tracking-wide"
        style={{
          color: 'var(--color-dark-text-muted)',
          borderBottom: '1px solid var(--color-dark-border)',
          background: 'var(--color-dark-card)',
        }}
      >
        Environment State
      </div>
      <div className="flex-1 overflow-auto" style={{ minHeight: 0, padding: '0.75rem' }}>
        <div className="space-y-3">
          {content}
          <details>
            <summary
              className="text-[10px] uppercase tracking-wide font-medium cursor-pointer select-none"
              style={{ color: 'var(--color-dark-text-muted)' }}
            >
              Raw environment state
            </summary>
            <div style={{ marginTop: '0.5rem' }}>
              <JsonSection label="Full state" value={fullState} />
            </div>
          </details>
        </div>
      </div>
    </div>
  )
}

function getRewardColor(value: number): string {
  if (value >= 0.8) return '#22c55e'
  if (value >= 0.4) return '#3b82f6'
  return 'var(--color-dark-text)'
}

function getScoreMetrics(sample: TraceSample) {
  return sample.score?.metrics ?? sample.rewards ?? []
}

function hasInput(sample: TraceSample): boolean {
  return sample.input != null && Object.keys(sample.input).length > 0
}

function hasMetadata(sample: TraceSample): boolean {
  return sample.metadata != null && Object.keys(sample.metadata).length > 0
}

function RunViewerContent({
  activeTab,
  sample,
  showWorkspaceSelection,
  checkedTurns,
  onToggleTurn,
  onMessageVisible,
  selectedTurn,
}: {
  activeTab: TabId
  sample: TraceSample
  showWorkspaceSelection: boolean
  checkedTurns: Set<number>
  onToggleTurn: (turn: number) => void
  onMessageVisible?: (messageIndex: number) => void
  selectedTurn: number
}) {
  const scoreMetrics = getScoreMetrics(sample)

  if (activeTab === 'conversation') {
    return (
      <ErrorBoundary label="ConversationView">
        <ConversationView
          sample={sample}
          selectedTurn={showWorkspaceSelection ? selectedTurn : undefined}
          onMessageVisible={onMessageVisible}
          checkedTurns={showWorkspaceSelection ? checkedTurns : undefined}
          onToggleTurn={showWorkspaceSelection ? onToggleTurn : undefined}
        />
      </ErrorBoundary>
    )
  }

  if (activeTab === 'summary') {
    return (
      <ErrorBoundary label="Summary">
        <div className="space-y-4">
          {sample.reward != null && (
            <SummaryField label="Reward">
              <span className="font-mono font-semibold text-sm" style={{ color: getRewardColor(sample.reward) }}>
                {sample.reward.toFixed(3)}
              </span>
            </SummaryField>
          )}

          {scoreMetrics.length > 0 && (
            <SummaryField label="Score breakdown">
              <div className="space-y-1">
                {scoreMetrics.map((metric, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <span style={{ color: 'var(--color-dark-text-secondary)' }}>{metric.name}</span>
                    <span
                      className="font-mono font-semibold"
                      style={{ color: getRewardColor(metric.value) }}
                    >
                      {metric.value.toFixed(3)}
                    </span>
                    {metric.weight !== 1.0 && (
                      <span className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>w={metric.weight}</span>
                    )}
                  </div>
                ))}
              </div>
            </SummaryField>
          )}

          {hasInput(sample) && (
            <SummaryField label="Input">
              <div className="space-y-2">
                {Object.entries(sample.input ?? {}).map(([key, value]) => (
                  <div key={key}>
                    <span className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--color-dark-text-muted)' }}>{key}: </span>
                    <span className="text-sm" style={{ color: 'var(--color-dark-text)' }}>
                      {typeof value === 'string' ? value : JSON.stringify(value)}
                    </span>
                  </div>
                ))}
              </div>
            </SummaryField>
          )}

          {hasMetadata(sample) && (
            <SummaryField label="Metadata">
              <div className="space-y-1">
                {Object.entries(sample.metadata ?? {}).map(([key, value]) => (
                  <div key={key} className="flex items-center gap-2 text-sm flex-wrap">
                    <span style={{ color: 'var(--color-dark-text-secondary)' }}>{key}</span>
                    <span className="font-mono" style={{ color: 'var(--color-dark-text)' }}>{String(value)}</span>
                  </div>
                ))}
              </div>
            </SummaryField>
          )}

          <SummaryField label="Completions">
            <p className="text-sm font-mono" style={{ color: 'var(--color-dark-text)' }}>
              {sample.trajectory.completions.length}
            </p>
          </SummaryField>
        </div>
      </ErrorBoundary>
    )
  }

  return (
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
  )
}

export function RunViewer({ runId, sampleId, evalName, liveStatus = null, onBack }: RunViewerProps) {
  const [activeTab, setActiveTab] = useState<TabId>('conversation')
  const [selectedTurn, setSelectedTurn] = useState(0)
  const [splitPct, setSplitPct] = useSplitPct()
  const [checkedTurns, setCheckedTurns] = useState<Set<number>>(new Set())
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const sampleState = useSampleViewData(runId, sampleId, liveStatus)

  const handleFitWidth = useCallback((contentWidthPx: number) => {
    if (!containerRef.current) return
    const totalWidth = containerRef.current.getBoundingClientRect().width
    if (totalWidth <= 0) return
    // Right panel needs: diff content + file sidebar (160px) + padding (24px)
    const rightNeeded = contentWidthPx + 160 + 24
    const fitLeftPct = Math.min(78, Math.max(10, Math.round((1 - rightNeeded / totalWidth) * 100)))
    setSplitPct(fitLeftPct)
  }, [setSplitPct])

  const handleToggleTurn = useCallback((turn: number) => {
    setCheckedTurns(prev => {
      const next = new Set(prev)
      if (next.has(turn)) next.delete(turn)
      else next.add(turn)
      return next
    })
  }, [])

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
    const messages = sampleState.kind === 'loaded' ? sampleState.sample.trajectory?.messages ?? [] : []
    let turn = 0
    for (let i = 0; i <= messageIndex && i < messages.length; i++) {
      if (messages[i]?.role === 'assistant') turn++
    }
    return Math.max(0, turn - 1)
  }, [sampleState])

  const sample = sampleState.kind === 'loaded' ? sampleState.sample : null
  const workspaceData = sampleState.kind === 'loaded' ? sampleState.workspaceData : null
  const checkedTurnsSorted = useMemo(() => [...checkedTurns].sort((a, b) => a - b), [checkedTurns])
  const hasWorkspace = workspaceData !== null && workspaceData.snapshots.some(s => Object.keys(s.files).length > 0)
  const hasEnvironmentPanel = sample !== null && hasEnvironmentState(sample)
  const hasTurnScopedSidePanel = hasWorkspace || hasEnvironmentPanel
  const plugin = sampleState.kind === 'loaded' && sampleState.source === 'artifact'
    ? resolvePlugin(evalName, sample)
    : null
  const hasRightPanel = hasWorkspace || plugin !== null || hasEnvironmentPanel

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
        {sampleState.kind === 'loaded' && sampleState.source === 'live' && (
          <span
            className="ml-auto inline-flex items-center gap-2 rounded px-3 py-2 text-xs font-medium"
            style={{
              color: '#22c55e',
              background: 'var(--color-dark-card)',
              border: '1px solid var(--color-dark-border)',
            }}
          >
            live
          </span>
        )}
        {sampleState.canExport && (
          <a
            href={getSampleHtmlExportUrl(runId, sampleId)}
            className="ml-auto inline-flex items-center gap-2 rounded px-3 py-2 text-xs font-medium transition-opacity hover:opacity-80"
            style={{
              color: 'var(--color-dark-text)',
              background: 'var(--color-dark-card)',
              border: '1px solid var(--color-dark-border)',
            }}
          >
            <Download className="h-3.5 w-3.5" />
            Export HTML
          </a>
        )}
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
  if (!hasRightPanel) {
    return (
      <div className="p-4 sm:p-6 max-w-5xl mx-auto" style={{ height: '100%', overflowY: 'auto', minHeight: 0 }}>
        {header}
        <div className="rounded p-4" style={{ background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>
          {sampleState.kind === 'loading' ? (
            <div className="flex items-center justify-center" style={{ height: 160 }}>
              <div className="rounded-full animate-spin" style={{ width: 24, height: 24, border: '2px solid var(--color-dark-border)', borderTopColor: 'var(--color-dark-text)' }} />
            </div>
          ) : sampleState.kind === 'error' ? (
            <p className="text-sm" style={{ color: '#ef4444' }}>{sampleState.error}</p>
          ) : sample ? (
            <RunViewerContent
              activeTab={activeTab}
              sample={sample}
              showWorkspaceSelection={false}
              checkedTurns={checkedTurns}
              onToggleTurn={handleToggleTurn}
              selectedTurn={selectedTurn}
            />
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
      {sampleState.kind === 'loading' ? (
        <div className="flex items-center justify-center rounded" style={{ height: 160, background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>
          <div className="rounded-full animate-spin" style={{ width: 24, height: 24, border: '2px solid var(--color-dark-border)', borderTopColor: 'var(--color-dark-text)' }} />
        </div>
      ) : sampleState.kind === 'error' ? (
        <p className="text-sm rounded p-4" style={{ color: '#ef4444', background: 'var(--color-dark-card)', border: '1px solid var(--color-dark-border)' }}>{sampleState.error}</p>
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
            <RunViewerContent
              activeTab={activeTab}
              sample={sample}
              showWorkspaceSelection={hasTurnScopedSidePanel}
              checkedTurns={checkedTurns}
              onToggleTurn={handleToggleTurn}
              onMessageVisible={hasTurnScopedSidePanel ? (idx) => setSelectedTurn(messageToTurn(idx)) : undefined}
              selectedTurn={selectedTurn}
            />
          </div>

          {/* Drag divider */}
          {hasRightPanel && (
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

          {/* Right panel: workspace or plugin */}
          {hasRightPanel && (
            <div style={{ flex: 1, minWidth: 0, height: '100%', overflow: 'hidden', borderRadius: 4, border: '1px solid var(--color-dark-border)', background: 'var(--color-dark-card)' }}>
              {hasWorkspace ? (
                <ErrorBoundary label="WorkspacePanel">
                  <WorkspacePanel
                    workspaceData={workspaceData!}
                    selectedTurn={selectedTurn}
                    checkedTurns={checkedTurnsSorted}
                    onFitWidth={handleFitWidth}
                  />
                </ErrorBoundary>
              ) : sample && hasEnvironmentPanel ? (
                <ErrorBoundary label="EnvironmentStatePanel">
                  <EnvironmentStatePanel
                    sample={sample}
                    selectedTurn={selectedTurn}
                    checkedTurns={checkedTurnsSorted}
                  />
                </ErrorBoundary>
              ) : plugin && sample ? (
                <ErrorBoundary label={plugin.label}>
                  {plugin.render({
                    sample,
                    runId,
                    selectedTurn,
                    checkedTurns: checkedTurnsSorted,
                  })}
                </ErrorBoundary>
              ) : null}
            </div>
          )}
        </div>
      ) : null}
      </div>{/* end content wrapper */}
    </div>
  )
}
