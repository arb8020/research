import { useState, useMemo, useRef, useCallback } from 'react'
import { ChevronRight, ChevronDown, File, Folder, Terminal, AlertTriangle } from 'lucide-react'
import type { WorkspaceData, WorkspaceSnapshot } from '../types'
import { DiffView } from './DiffView'
import { File as DiffsFile } from '@pierre/diffs/react'

type PanelMode = 'state' | 'diff' | 'split'

const SPLIT_HEIGHT_KEY = 'workspace-split-height-pct'
const DEFAULT_SPLIT_HEIGHT = 50

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildFileTree(files: Record<string, string>): TreeNode[] {
  const root: Record<string, TreeNode> = {}

  for (const path of Object.keys(files)) {
    const parts = path.split('/')
    let current = root
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const isFile = i === parts.length - 1
      if (!current[part]) {
        current[part] = { name: part, path: parts.slice(0, i + 1).join('/'), isFile, children: {} }
      }
      if (!isFile) current = current[part].children
    }
  }

  function flatten(nodes: Record<string, TreeNode>): TreeNode[] {
    return Object.values(nodes).sort((a, b) => {
      if (a.isFile !== b.isFile) return a.isFile ? 1 : -1
      return a.name.localeCompare(b.name)
    })
  }

  function build(nodes: Record<string, TreeNode>): TreeNode[] {
    return flatten(nodes).map(n => ({
      ...n,
      children_list: n.isFile ? [] : build(n.children),
    }))
  }

  return build(root)
}

interface TreeNode {
  name: string
  path: string
  isFile: boolean
  children: Record<string, TreeNode>
  children_list?: TreeNode[]
}

// ── FileTree ──────────────────────────────────────────────────────────────────

function FileTree({
  files,
  selectedFile,
  onSelect,
  modifiedFiles,
}: {
  files: Record<string, string>
  selectedFile: string | null
  onSelect: (path: string) => void
  modifiedFiles: Set<string>
}) {
  const [open, setOpen] = useState<Set<string>>(new Set())
  const tree = useMemo(() => buildFileTree(files), [files])

  function toggle(path: string) {
    setOpen(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  function renderNodes(nodes: TreeNode[], depth = 0): React.ReactNode {
    return nodes.map(node => {
      if (node.isFile) {
        const isSelected = selectedFile === node.path
        const isModified = modifiedFiles.has(node.path)
        return (
          <div
            key={node.path}
            onClick={() => onSelect(node.path)}
            className="flex items-center gap-1 cursor-pointer rounded px-1 py-0.5 text-xs font-mono select-none"
            style={{
              paddingLeft: `${depth * 12 + 4}px`,
              background: isSelected ? 'rgba(59,130,246,0.18)' : undefined,
              color: isSelected ? '#93c5fd' : isModified ? '#fbbf24' : 'var(--color-dark-text-secondary)',
            }}
          >
            <File className="h-3 w-3 flex-shrink-0" />
            <span className="truncate">{node.name}</span>
            {isModified && <span style={{ color: '#fbbf24', fontSize: 8, marginLeft: 2 }}>●</span>}
          </div>
        )
      }
      const isOpen = open.has(node.path)
      return (
        <div key={node.path}>
          <div
            onClick={() => toggle(node.path)}
            className="flex items-center gap-1 cursor-pointer rounded px-1 py-0.5 text-xs font-mono select-none"
            style={{
              paddingLeft: `${depth * 12 + 4}px`,
              color: 'var(--color-dark-text-muted)',
            }}
          >
            {isOpen ? <ChevronDown className="h-3 w-3 flex-shrink-0" /> : <ChevronRight className="h-3 w-3 flex-shrink-0" />}
            <Folder className="h-3 w-3 flex-shrink-0" />
            <span>{node.name}</span>
          </div>
          {isOpen && renderNodes(node.children_list ?? [], depth + 1)}
        </div>
      )
    })
  }

  return (
    <div className="overflow-y-auto h-full py-1">
      {renderNodes(tree)}
    </div>
  )
}

// ── BashHistory ───────────────────────────────────────────────────────────────

function BashHistory({ entries }: { entries: WorkspaceSnapshot['bash_history'] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  if (entries.length === 0) return (
    <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
      no commands yet
    </div>
  )

  return (
    <div className="overflow-y-auto h-full font-mono text-xs">
      {entries.map((entry, i) => {
        const isExp = expanded.has(i)
        const hasOutput = entry.stdout || entry.stderr
        return (
          <div
            key={i}
            className="border-b"
            style={{ borderColor: '#1e1e1e' }}
          >
            <div
              className="flex items-start gap-2 px-2 py-1"
              style={{ cursor: hasOutput ? 'pointer' : 'default' }}
              onClick={() => {
                if (!hasOutput) return
                setExpanded(prev => {
                  const next = new Set(prev)
                  if (next.has(i)) next.delete(i); else next.add(i)
                  return next
                })
              }}
            >
              <span className="flex-shrink-0 mt-0.5" style={{ color: entry.exit_code !== 0 ? '#ef4444' : '#22c55e', fontSize: 9 }}>
                {entry.exit_code !== 0 ? '✗' : '✓'}
              </span>
              {entry.uncertain_fs_effects && (
                <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" style={{ color: '#f59e0b' }} />
              )}
              <span className="flex-1 truncate" style={{ color: '#e2e8f0' }}>{entry.cmd}</span>
              {hasOutput && (
                <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 9 }}>
                  {isExp ? '▲' : '▼'}
                </span>
              )}
            </div>
            {isExp && hasOutput && (
              <div
                className="px-2 pb-1.5 whitespace-pre-wrap overflow-x-auto"
                style={{ color: '#94a3b8', background: '#0a0a0a', fontSize: 10, maxHeight: 120 }}
              >
                {entry.stdout || entry.stderr}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── WorkspacePanel (main) ─────────────────────────────────────────────────────

export function WorkspacePanel({
  workspaceData,
  selectedTurn,
  checkedTurns,
  onFitWidth,
}: {
  workspaceData: WorkspaceData
  selectedTurn: number
  checkedTurns: number[]
  onFitWidth?: (contentWidthPx: number) => void
}) {
  const { snapshots, line_history, source } = workspaceData

  // Default to diff mode showing full run (turn 0 → last turn)
  const [mode, setMode] = useState<PanelMode>('diff')
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  // Resizable split: top panel height as % of total (persisted)
  const [splitHeightPct, setSplitHeightPct] = useState(() => {
    try {
      return Number(localStorage.getItem(SPLIT_HEIGHT_KEY)) || DEFAULT_SPLIT_HEIGHT
    } catch {
      return DEFAULT_SPLIT_HEIGHT
    }
  })
  const splitContainerRef = useRef<HTMLDivElement>(null)
  const splitDragging = useRef(false)

  const onSplitDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    splitDragging.current = true
    const onMove = (ev: MouseEvent) => {
      if (!splitDragging.current || !splitContainerRef.current) return
      const rect = splitContainerRef.current.getBoundingClientRect()
      const pct = Math.min(80, Math.max(20, ((ev.clientY - rect.top) / rect.height) * 100))
      const rounded = Math.round(pct)
      setSplitHeightPct(rounded)
      try {
        localStorage.setItem(SPLIT_HEIGHT_KEY, String(rounded))
      } catch {
        return
      }
    }
    const onUp = () => { splitDragging.current = false; window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  // Find snapshot at or before a given turn
  function snapshotAt(turn: number): WorkspaceSnapshot {
    const candidates = snapshots.filter(s => s.turn <= turn)
    return candidates.length > 0 ? candidates[candidates.length - 1] : snapshots[0]
  }

  // Snapshot just before turn T: the snapshot at end of turn T-1
  function snapshotBefore(turn: number): WorkspaceSnapshot {
    if (turn === 0) return snapshots[0]
    const candidates = snapshots.filter(s => s.turn < turn)
    return candidates.length > 0 ? candidates[candidates.length - 1] : snapshots[0]
  }

  // Derive diff range from checkedTurns:
  //   0 checked → snapshot[0] to last snapshot (full run)
  //   1 checked (turn T) → snapshot just before T to snapshot at T (before/after that turn)
  //   2+ checked → snapshot before min-turn to snapshot at max-turn
  const { snapshotA, snapshotB } = (() => {
    const last = snapshots[snapshots.length - 1]
    if (checkedTurns.length === 0) {
      return { snapshotA: snapshots[0], snapshotB: last }
    }
    if (checkedTurns.length === 1) {
      const t = checkedTurns[0]
      return { snapshotA: snapshotBefore(t), snapshotB: snapshotAt(t) }
    }
    const sorted = [...checkedTurns].sort((a, b) => a - b)
    return {
      snapshotA: snapshotBefore(sorted[0]),
      snapshotB: snapshotAt(sorted[sorted.length - 1]),
    }
  })()

  // State mode uses selectedTurn (driven by clicking conversation messages)
  const snapshot = snapshotAt(selectedTurn)

  const modifiedFiles = useMemo(() => new Set(Object.keys(line_history)), [line_history])
  const preferredDiffFile = Object.keys(snapshotB.files).find(
    f => (snapshotA.files[f] ?? '') !== (snapshotB.files[f] ?? '')
  ) ?? Object.keys(snapshotB.files)[0] ?? null
  const preferredStateFile = Object.keys(snapshot.files)[0] ?? null
  const activeFileSet = mode === 'diff' ? snapshotB.files : snapshot.files
  const selectedFileForMode = selectedFile && selectedFile in activeFileSet
    ? selectedFile
    : mode === 'diff'
    ? preferredDiffFile
    : preferredStateFile
  const fileContents = selectedFileForMode ? snapshot.files[selectedFileForMode] ?? '' : ''

  return (
    <div
      className="flex flex-col h-full rounded overflow-hidden"
      style={{ border: '1px solid var(--color-dark-border)', background: 'var(--color-dark-card)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-1.5 text-xs flex-shrink-0"
        style={{ borderBottom: '1px solid var(--color-dark-border)', color: 'var(--color-dark-text-muted)' }}
      >
        <div className="flex items-center gap-2">
          <Terminal className="h-3 w-3" />
          <span className="font-mono">workspace</span>
          {mode === 'state' ? (
            <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 10 }}>turn {snapshot.turn}</span>
          ) : mode === 'diff' ? (
            <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 10 }}>
              {snapshotA.turn} → {snapshotB.turn}
            </span>
          ) : (
            <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 10 }}>
              turn {snapshot.turn} / {snapshotA.turn} → {snapshotB.turn}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {source === 'reconstructed' && (
            <span style={{ color: '#f59e0b', fontSize: 9 }}>⚠ reconstructed</span>
          )}
          {/* Fit width button — only in diff/split mode */}
          {mode !== 'state' && (
            <button
              onClick={() => {
                if (!onFitWidth) return
                const dc = document.querySelector('diffs-container') as HTMLElement | null
                if (!dc) return
                // Temporarily force dc narrow so [data-code] overflows and scrollWidth = true content width
                const saved = dc.style.width
                dc.style.width = '1px'
                const shadow = dc.shadowRoot
                const dataCode = shadow?.querySelector('[data-code]') as HTMLElement | null
                const contentWidthPx = dataCode?.scrollWidth ?? 600
                dc.style.width = saved
                onFitWidth(contentWidthPx)
              }}
              title="Expand panel to fit diff width"
              className="font-mono"
              style={{ fontSize: 10, color: 'var(--color-dark-text-muted)', padding: '1px 4px', border: '1px solid var(--color-dark-border)', borderRadius: 2 }}
            >
              ↔
            </button>
          )}
          {/* [State | Diff | Split] toggle */}
          <div className="flex rounded overflow-hidden" style={{ border: '1px solid var(--color-dark-border)', fontSize: 10 }}>
            {(['state', 'diff', 'split'] as PanelMode[]).map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className="px-2 py-0.5 font-mono capitalize"
                style={{
                  background: mode === m ? 'var(--color-dark-text)' : 'transparent',
                  color: mode === m ? 'var(--color-dark-bg)' : 'var(--color-dark-text-muted)',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main area */}
      <div ref={splitContainerRef} className="flex flex-col flex-1 overflow-hidden" style={{ minHeight: 0 }}>
        {mode === 'diff' ? (
          <DiffView
            snapshotA={snapshotA}
            snapshotB={snapshotB}
            selectedFile={selectedFileForMode}
            onSelectFile={setSelectedFile}
          />
        ) : mode === 'state' ? (
          <div className="flex flex-1 overflow-hidden" style={{ minHeight: 0 }}>
            {/* File tree (left) */}
            <div
              className="flex-shrink-0 overflow-hidden"
              style={{ width: 160, borderRight: '1px solid var(--color-dark-border)', background: '#111' }}
            >
              <FileTree
                files={snapshot.files}
                selectedFile={selectedFileForMode}
                onSelect={setSelectedFile}
                modifiedFiles={modifiedFiles}
              />
            </div>
            {/* File viewer (right) */}
            <div className="flex-1 overflow-auto" style={{ minWidth: 0, minHeight: 0 }}>
              {selectedFileForMode ? (
                <DiffsFile
                  file={{ name: selectedFileForMode, contents: fileContents }}
                  options={{ theme: 'pierre-dark' }}
                  style={{ width: '100%', display: 'block' }}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
                  select a file
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Split mode: State on top, Diff on bottom, resizable */
          <>
            {/* State panel (top) */}
            <div className="flex overflow-hidden flex-shrink-0" style={{ height: `${splitHeightPct}%`, minHeight: 0 }}>
              <div
                className="flex-shrink-0 overflow-hidden"
                style={{ width: 160, borderRight: '1px solid var(--color-dark-border)', background: '#111' }}
              >
                <FileTree
                  files={snapshot.files}
                  selectedFile={selectedFileForMode}
                  onSelect={setSelectedFile}
                  modifiedFiles={modifiedFiles}
                />
              </div>
              <div className="flex-1 overflow-auto" style={{ minWidth: 0, minHeight: 0 }}>
                {selectedFileForMode ? (
                  <DiffsFile
                    file={{ name: selectedFileForMode, contents: fileContents }}
                    options={{ theme: 'pierre-dark' }}
                    style={{ width: '100%', display: 'block' }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
                    select a file
                  </div>
                )}
              </div>
            </div>
            {/* Horizontal drag divider */}
            <div
              onMouseDown={onSplitDividerMouseDown}
              style={{
                height: 5,
                flexShrink: 0,
                cursor: 'row-resize',
                background: 'transparent',
                transition: 'background 100ms',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-dark-border)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />
            {/* Diff panel (bottom) */}
            <DiffView
              snapshotA={snapshotA}
              snapshotB={snapshotB}
              selectedFile={selectedFileForMode}
              onSelectFile={setSelectedFile}
            />
          </>
        )}
      </div>

      {/* Bash history (bottom) */}
      <div
        style={{
          height: 140,
          borderTop: '1px solid var(--color-dark-border)',
          background: '#0d0d0d',
          flexShrink: 0,
        }}
      >
        <div
          className="px-2 py-0.5 text-[10px] uppercase tracking-wide font-medium flex-shrink-0"
          style={{ color: 'var(--color-dark-text-muted)', borderBottom: '1px solid #1e1e1e' }}
        >
          bash history ({(mode === 'diff' ? snapshotB : snapshot).bash_history.length})
        </div>
        <div style={{ height: 'calc(100% - 20px)' }}>
          <BashHistory entries={(mode === 'diff' ? snapshotB : snapshot).bash_history} />
        </div>
      </div>
    </div>
  )
}
