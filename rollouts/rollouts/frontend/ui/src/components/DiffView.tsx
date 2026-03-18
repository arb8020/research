import { useMemo, useEffect } from 'react'
import { FileDiff } from '@pierre/diffs/react'
import { parseDiffFromFile } from '@pierre/diffs'
import type { WorkspaceSnapshot } from '../types'

// Force diffs-container custom element to fill its flex parent
let diffStylesInjected = false
function ensureDiffStyles() {
  if (diffStylesInjected) return
  diffStylesInjected = true
  const el = document.createElement('style')
  el.textContent = 'diffs-container { display: block; width: 100%; }'
  document.head.appendChild(el)
}

interface DiffViewProps {
  snapshotA: WorkspaceSnapshot         // "from" state
  snapshotB: WorkspaceSnapshot         // "to" state
  selectedFile: string | null
  onSelectFile: (path: string) => void
}

function buildFileDiff(name: string, oldContents: string, newContents: string) {
  return parseDiffFromFile(
    { name, contents: oldContents },
    { name, contents: newContents },
  )
}

export function DiffView({ snapshotA, snapshotB, selectedFile, onSelectFile }: DiffViewProps) {
  useEffect(() => { ensureDiffStyles() }, [])

  // Compute per-file diffs, only for files that exist in either snapshot
  const diffs = useMemo(() => {
    const allFiles = new Set([
      ...Object.keys(snapshotA.files),
      ...Object.keys(snapshotB.files),
    ])
    const result: Array<{ name: string; diff: ReturnType<typeof buildFileDiff>; hasChanges: boolean }> = []
    for (const name of Array.from(allFiles).sort()) {
      const oldContents = snapshotA.files[name] ?? ''
      const newContents = snapshotB.files[name] ?? ''
      const diff = buildFileDiff(name, oldContents, newContents)
      result.push({ name, diff, hasChanges: oldContents !== newContents })
    }
    return result
  }, [snapshotA, snapshotB])

  const changedFiles = diffs.filter(d => d.hasChanges)
  const unchangedFiles = diffs.filter(d => !d.hasChanges)

  if (diffs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
        no files
      </div>
    )
  }

  if (changedFiles.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
        no changes between turn {snapshotA.turn} and turn {snapshotB.turn}
      </div>
    )
  }

  const displayDiff = selectedFile
    ? diffs.find(d => d.name === selectedFile) ?? changedFiles[0]
    : changedFiles[0]

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0, minWidth: 0 }}>
      {/* File list sidebar */}
      <div
        className="flex-shrink-0 overflow-y-auto py-1"
        style={{ width: 160, borderRight: '1px solid var(--color-dark-border)', background: '#111' }}
      >
        {changedFiles.length > 0 && (
          <div className="px-2 py-0.5 text-[9px] uppercase tracking-wide font-medium" style={{ color: '#f59e0b' }}>
            changed ({changedFiles.length})
          </div>
        )}
        {changedFiles.map(({ name }) => (
          <div
            key={name}
            onClick={() => onSelectFile(name)}
            className="px-2 py-0.5 text-xs font-mono cursor-pointer rounded truncate"
            style={{
              color: selectedFile === name ? '#93c5fd' : '#f59e0b',
              background: selectedFile === name ? 'rgba(59,130,246,0.18)' : undefined,
            }}
          >
            {name.split('/').pop()}
          </div>
        ))}
        {unchangedFiles.length > 0 && (
          <>
            <div className="px-2 py-0.5 mt-1 text-[9px] uppercase tracking-wide font-medium" style={{ color: 'var(--color-dark-text-muted)' }}>
              unchanged ({unchangedFiles.length})
            </div>
            {unchangedFiles.map(({ name }) => (
              <div
                key={name}
                onClick={() => onSelectFile(name)}
                className="px-2 py-0.5 text-xs font-mono cursor-pointer rounded truncate"
                style={{
                  color: selectedFile === name ? '#93c5fd' : 'var(--color-dark-text-muted)',
                  background: selectedFile === name ? 'rgba(59,130,246,0.18)' : undefined,
                }}
              >
                {name.split('/').pop()}
              </div>
            ))}
          </>
        )}
      </div>

      {/* Diff content */}
      <div style={{ flex: 1, overflow: 'auto', background: '#0d1117', minWidth: 0, minHeight: 0, height: '100%' }}>
        {displayDiff.hasChanges ? (
          <FileDiff
            fileDiff={displayDiff.diff}
            options={{
              theme: 'github-dark',
              diffStyle: 'unified',
            }}
            style={{ width: '100%', display: 'block' }}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
            no changes in {displayDiff.name}
          </div>
        )}
      </div>
    </div>
  )
}
