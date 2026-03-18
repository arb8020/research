import { useMemo } from 'react'
import { FileDiff, File as DiffsFile } from '@pierre/diffs/react'
import { parseDiffFromFile } from '@pierre/diffs'
import type { WorkspaceSnapshot } from '../types'

// Static styles applied once to document head — controls diffs-container layout and theme.
const docStyle = document.createElement('style')
docStyle.textContent = `
  diffs-container {
    display: block;
    width: 100%;
    /* Force dark mode so light-dark() resolves to dark variants */
    color-scheme: dark;
    /* Typography */
    --diffs-font-size: 12px;
    --diffs-line-height: 1.5;
    --diffs-font-family: "IBM Plex Mono", monospace;
    --diffs-header-font-family: "IBM Plex Mono", monospace;
    --diffs-gap-inline: 0px;
    --diffs-gap-block: 0px;
  }
`
document.head.appendChild(docStyle)

// Static overrides injected into diffs-container's shadow root whenever it appears.
// Uses MutationObserver so injection survives the library tearing down and recreating
// the shadow DOM on file change — no useEffect, no setTimeout, no React lifecycle dependency.
const SHADOW_OVERRIDE_ID = 'rollouts-overrides'
const SHADOW_OVERRIDE_CSS = `
  [data-separator-content] { border-radius: 0 !important; }
  [data-expand-button] { border-radius: 0 !important; }
  [data-separator-wrapper] { border-radius: 0 !important; }
  /* Kill pill radius from @supports (width: 1cqi) block */
  [data-unified] [data-separator='line-info'] [data-separator-wrapper] [data-separator-content] { border-radius: 0 !important; }
  [data-gutter] [data-separator='line-info'] [data-separator-content] { border-radius: 0 !important; }
  [data-separator='line-info'] [data-separator-wrapper] [data-expand-both],
  [data-separator='line-info'] [data-separator-wrapper] [data-expand-down],
  [data-separator='line-info'] [data-separator-wrapper] [data-expand-up] { border-radius: 0 !important; }
  /* More breathing room below the file header */
  [data-diffs-header] { padding-block: 10px 14px; }
  /* Gap between file icon and filename */
  [data-header-content] { gap: 8px; }
`

// TODO: We don't know whether @pierre/diffs recreates the diffs-container element
// on file change (triggering bodyObserver) or mutates the shadow root in place
// (triggering the inner shadow observer). We're defending against both simultaneously.
// Add console.logs on each path to find out which actually fires, then remove the
// redundant observer.
function injectIntoShadow(el: Element) {
  const shadow = (el as HTMLElement).shadowRoot
  if (!shadow) return
  if (shadow.querySelector(`#${SHADOW_OVERRIDE_ID}`)) return
  const style = document.createElement('style')
  style.id = SHADOW_OVERRIDE_ID
  style.textContent = SHADOW_OVERRIDE_CSS
  shadow.appendChild(style)
  // Re-inject if the library wipes shadow root children without removing the element.
  // TODO: may be redundant with bodyObserver if the library always recreates the element.
  new MutationObserver(() => {
    if (!shadow.querySelector(`#${SHADOW_OVERRIDE_ID}`)) {
      const s = document.createElement('style')
      s.id = SHADOW_OVERRIDE_ID
      s.textContent = SHADOW_OVERRIDE_CSS
      shadow.appendChild(s)
    }
  }).observe(shadow, { childList: true })
}

// Watch document for diffs-container being added, then inject into its shadow root.
// TODO: may be redundant with the inner shadow observer in injectIntoShadow.
const bodyObserver = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    for (const node of Array.from(mutation.addedNodes)) {
      if (!(node instanceof Element)) continue
      if (node.tagName === 'DIFFS-CONTAINER') injectIntoShadow(node)
      node.querySelectorAll('diffs-container').forEach(injectIntoShadow)
    }
  }
})

// document.body may not exist at module parse time — defer until DOM ready.
function startObserving() {
  bodyObserver.observe(document.body, { childList: true, subtree: true })
  // Handle case where diffs-container already exists in the DOM
  document.querySelectorAll('diffs-container').forEach(injectIntoShadow)
}

if (document.body) {
  startObserving()
} else {
  document.addEventListener('DOMContentLoaded', startObserving)
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

      {/* Diff/file content */}
      <div style={{ flex: 1, overflow: 'auto', background: '#0d1117', minWidth: 0, minHeight: 0, height: '100%' }}>
        {displayDiff.hasChanges ? (
          <FileDiff
            fileDiff={displayDiff.diff}
            options={{ theme: 'pierre-dark', diffStyle: 'unified' }}
            style={{ width: '100%', display: 'block' }}
          />
        ) : (
          <DiffsFile
            file={{ name: displayDiff.name, contents: snapshotB.files[displayDiff.name] ?? snapshotA.files[displayDiff.name] ?? '' }}
            options={{ theme: 'pierre-dark' }}
            style={{ width: '100%', display: 'block' }}
          />
        )}
      </div>
    </div>
  )
}
