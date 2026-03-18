import { useState, useMemo, useRef, useEffect } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { ChevronRight, ChevronDown, File, Folder, Terminal, AlertTriangle } from 'lucide-react'
import type { WorkspaceData, WorkspaceSnapshot, LineEdit } from '../types'

// ── Helpers ───────────────────────────────────────────────────────────────────

function detectLanguage(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  const map: Record<string, string> = {
    py: 'python', js: 'javascript', ts: 'typescript', tsx: 'tsx',
    jsx: 'jsx', sh: 'bash', bash: 'bash', md: 'markdown',
    json: 'json', yaml: 'yaml', yml: 'yaml', toml: 'toml',
    rs: 'rust', go: 'go', c: 'c', cpp: 'cpp', h: 'c',
  }
  return map[ext] ?? 'text'
}

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

// ── LineBlamePopover ──────────────────────────────────────────────────────────

function LineBlamePopover({
  edits,
  onJumpToMessage,
  onClose,
}: {
  edits: LineEdit[]
  onJumpToMessage: (messageIndex: number) => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [onClose])

  return (
    <div
      ref={ref}
      className="absolute z-50 rounded shadow-lg text-xs"
      style={{
        right: 0,
        top: '100%',
        minWidth: 280,
        maxWidth: 400,
        background: '#18181b',
        border: '1px solid var(--color-dark-border)',
        maxHeight: 240,
        overflowY: 'auto',
      }}
    >
      <div className="px-2 py-1 font-semibold" style={{ color: 'var(--color-dark-text-muted)', borderBottom: '1px solid var(--color-dark-border)' }}>
        {edits.length} edit{edits.length !== 1 ? 's' : ''} touched this line
      </div>
      {[...edits].reverse().map((edit, i) => (
        <div
          key={i}
          className="px-2 py-1.5 cursor-pointer hover:opacity-80"
          style={{ borderBottom: i < edits.length - 1 ? '1px solid #27272a' : undefined }}
          onClick={() => { onJumpToMessage(edit.message_index); onClose() }}
        >
          <div className="flex items-center gap-2 mb-0.5">
            <span
              className="rounded px-1 py-0.5 font-mono"
              style={{
                background: edit.type === 'patch' ? '#1e3a2f' : edit.type === 'write' ? '#1e2a3f' : '#2a1e1e',
                color: edit.type === 'patch' ? '#22c55e' : edit.type === 'write' ? '#93c5fd' : '#fca5a5',
              }}
            >
              {edit.type}
            </span>
            <span style={{ color: 'var(--color-dark-text-muted)' }}>turn {edit.turn}</span>
            <span className="ml-auto" style={{ color: '#3b82f6', textDecoration: 'underline' }}>→ jump</span>
          </div>
          {edit.diff && (
            <div className="font-mono truncate" style={{ color: '#a1a1aa', fontSize: 10 }}>
              {edit.diff.slice(0, 80)}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ── FileViewer ────────────────────────────────────────────────────────────────

function FileViewer({
  filename,
  contents,
  lineHistory,
  onJumpToMessage,
}: {
  filename: string
  contents: string
  lineHistory: Record<string, LineEdit[]>
  onJumpToMessage: (messageIndex: number) => void
}) {
  const [blameLine, setBlameLine] = useState<number | null>(null)

  const editedLines = useMemo(() => {
    return new Set(Object.keys(lineHistory).map(Number))
  }, [lineHistory])

  return (
    <div className="relative h-full overflow-auto" style={{ background: '#0d1117' }}>
      <div className="text-[10px] font-mono px-2 py-1 sticky top-0 flex items-center gap-2" style={{ background: '#161b22', borderBottom: '1px solid #30363d', color: '#8b949e' }}>
        <span>{filename}</span>
        {Object.keys(lineHistory).length > 0 && (
          <span style={{ color: '#fbbf24' }}>
            {Object.keys(lineHistory).length} edited lines
          </span>
        )}
      </div>
      <SyntaxHighlighter
        language={detectLanguage(filename)}
        style={vscDarkPlus}
        showLineNumbers
        wrapLines
        lineNumberStyle={{ minWidth: '2.5em', paddingRight: '0.5em', color: '#484f58', userSelect: 'none' }}
        customStyle={{ margin: 0, padding: '0.5rem 0', background: 'transparent', fontSize: '11px', lineHeight: '1.5' }}
        lineProps={(lineNumber) => {
          const hasEdit = editedLines.has(lineNumber)
          const isBlameOpen = blameLine === lineNumber
          return {
            style: {
              display: 'block',
              cursor: hasEdit ? 'pointer' : 'default',
              background: isBlameOpen
                ? 'rgba(59,130,246,0.15)'
                : hasEdit
                ? 'rgba(251,191,36,0.06)'
                : undefined,
              position: 'relative' as const,
            },
            onClick: hasEdit ? () => setBlameLine(blameLine === lineNumber ? null : lineNumber) : undefined,
          }
        }}
      >
        {contents}
      </SyntaxHighlighter>

      {/* Blame popover rendered outside syntax highlighter */}
      {blameLine !== null && lineHistory[String(blameLine)] && (
        <div
          className="absolute"
          style={{ top: `${(blameLine - 1) * 18 + 28}px`, right: '8px' }}
        >
          <LineBlamePopover
            edits={lineHistory[String(blameLine)]}
            onJumpToMessage={onJumpToMessage}
            onClose={() => setBlameLine(null)}
          />
        </div>
      )}
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
  onJumpToMessage,
}: {
  workspaceData: WorkspaceData
  selectedTurn: number
  onJumpToMessage: (messageIndex: number) => void
}) {
  const { snapshots, line_history, source } = workspaceData

  // Find snapshot for current turn (or nearest prior)
  const snapshot = useMemo(() => {
    const candidates = snapshots.filter(s => s.turn <= selectedTurn)
    return candidates.length > 0 ? candidates[candidates.length - 1] : snapshots[0]
  }, [snapshots, selectedTurn])

  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  // Auto-select first file
  useEffect(() => {
    if (!selectedFile && snapshot) {
      const files = Object.keys(snapshot.files)
      if (files.length > 0) setSelectedFile(files[0])
    }
  }, [snapshot])

  const modifiedFiles = useMemo(() => new Set(Object.keys(line_history)), [line_history])

  if (!snapshot) return null

  const fileContents = selectedFile ? snapshot.files[selectedFile] ?? '' : ''
  const fileLineHistory = selectedFile ? (line_history[selectedFile] ?? {}) : {}

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
          <span style={{ color: 'var(--color-dark-text-muted)', fontSize: 10 }}>
            turn {snapshot.turn}
          </span>
        </div>
        {source === 'reconstructed' && (
          <span style={{ color: '#f59e0b', fontSize: 9 }}>⚠ reconstructed</span>
        )}
      </div>

      {/* Main area: file tree + file viewer */}
      <div className="flex flex-1 overflow-hidden" style={{ minHeight: 0 }}>
        {/* File tree (left) */}
        <div
          className="flex-shrink-0 overflow-hidden"
          style={{
            width: 160,
            borderRight: '1px solid var(--color-dark-border)',
            background: '#111',
          }}
        >
          <FileTree
            files={snapshot.files}
            selectedFile={selectedFile}
            onSelect={setSelectedFile}
            modifiedFiles={modifiedFiles}
          />
        </div>

        {/* File viewer (right) */}
        <div className="flex-1 overflow-hidden relative">
          {selectedFile ? (
            <FileViewer
              filename={selectedFile}
              contents={fileContents}
              lineHistory={fileLineHistory}
              onJumpToMessage={onJumpToMessage}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
              select a file
            </div>
          )}
        </div>
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
          bash history ({snapshot.bash_history.length})
        </div>
        <div style={{ height: 'calc(100% - 20px)' }}>
          <BashHistory entries={snapshot.bash_history} />
        </div>
      </div>
    </div>
  )
}
