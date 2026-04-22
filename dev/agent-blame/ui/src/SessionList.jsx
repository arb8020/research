// Sessions view for the left sidebar.
//
// Each row is a session (source + session_id). Collapsed shows a summary
// line (source, session-8, total lines, file count). Expanded reveals the
// files this session touched, click to open in the center panel.
//
// Session header click opens the transcript in the right panel, scrolled
// to the end (no specific edit anchor).
//
// Low-signal noise suppression: provenance cross-file matches mean a
// large session "touches" many files with just 1-2 matching lines each.
// We show only files with ≥ MIN_FILE_LINES by default, with a toggle
// to reveal all.

import { useMemo, useState } from 'react'

const MIN_FILE_LINES_DEFAULT = 3

const SORTS = {
  lines:  { label: 'lines',  keyFn: s => -(s.total_lines || 0) },
  recent: { label: 'recent', keyFn: s => s.latest_edit ? -new Date(s.latest_edit).getTime() : 0 },
  files:  { label: 'files',  keyFn: s => -(s.files?.length || 0) },
}

function colorFor(source, sessionId) {
  let h = 0
  const s = `${source}:${sessionId}`
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  const PALETTE = [
    '#4a9eff', '#00ec7e', '#f35b00', '#b685ff', '#ffb84a',
    '#ff6384', '#4aeaff', '#a3e635', '#f472b6', '#fbbf24',
    '#60a5fa', '#fb923c',
  ]
  return PALETTE[Math.abs(h) % PALETTE.length]
}

function formatLatest(iso) {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch { return '' }
}

function SessionRow({ s, expanded, onToggle, onOpenSession, onOpenFile, selectedPath, minFileLines, showAll }) {
  const color = colorFor(s.source, s.session_id)
  const visibleFiles = showAll ? s.files : s.files.filter(f => f.lines >= minFileLines)
  const hiddenCount = s.files.length - visibleFiles.length

  return (
    <div>
      <div
        onClick={onToggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 10px',
          cursor: 'pointer',
          background: 'transparent',
        }}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-elevated)'}
        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
      >
        <span style={{
          fontSize: 8, color: 'var(--text-disabled)', width: 8, textAlign: 'center',
          transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
          transition: 'transform 80ms',
        }}>▶</span>
        <span style={{
          width: 6, height: 6, borderRadius: 3, background: color, flexShrink: 0,
        }} />
        <span style={{ color: 'var(--text-primary)', fontSize: 11.5 }}>
          {s.source}
        </span>
        <span style={{ color: 'var(--text-secondary)', fontSize: 11.5 }}>
          {s.session_id.slice(0, 8)}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-disabled)' }}>
          {s.total_lines}L · {s.files.length}f
        </span>
      </div>
      <div style={{
        fontSize: 10, color: 'var(--text-disabled)',
        paddingLeft: 30, paddingBottom: expanded ? 0 : 4,
      }}>
        {formatLatest(s.latest_edit)}
        <span
          onClick={e => { e.stopPropagation(); onOpenSession(s) }}
          style={{ marginLeft: 8, color: 'var(--text-secondary)', cursor: 'pointer', textDecoration: 'underline' }}
        >
          open session →
        </span>
      </div>
      {expanded && visibleFiles.map(f => {
        const isSel = selectedPath === f.path
        return (
          <div
            key={f.path}
            onClick={() => onOpenFile(f.path)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '3px 10px',
              paddingLeft: 30,
              cursor: 'pointer',
              background: isSel ? 'var(--bg-elevated)' : 'transparent',
              borderLeft: isSel ? '2px solid var(--text-green)' : '2px solid transparent',
              fontSize: 11,
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'var(--bg-elevated)' }}
            onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = 'transparent' }}
          >
            <span style={{
              flex: 1, overflow: 'hidden', textOverflow: 'ellipsis',
              color: isSel ? 'var(--text-primary)' : 'var(--text-secondary)',
            }}>
              {f.path}
            </span>
            <span style={{ color: 'var(--text-disabled)', fontSize: 10 }}>
              {f.lines}
            </span>
          </div>
        )
      })}
      {expanded && hiddenCount > 0 && (
        <div style={{
          fontSize: 10, color: 'var(--text-disabled)',
          paddingLeft: 30, paddingBottom: 4, paddingTop: 2,
        }}>
          {hiddenCount} more files with &lt; {minFileLines} lines (low-signal matches hidden)
        </div>
      )}
    </div>
  )
}

export function SessionList({ sessions, selectedPath, onOpenSession, onOpenFile }) {
  const [expanded, setExpanded] = useState(new Set())
  const [showAll, setShowAll] = useState(false)
  const [sortKey, setSortKey] = useState(
    () => localStorage.getItem('ab-session-sort') || 'lines'
  )

  const sorted = useMemo(() => {
    if (!sessions) return null
    const sort = SORTS[sortKey] || SORTS.lines
    // Slice so we don't mutate the caller's array, then stable sort.
    return [...sessions].sort((a, b) => {
      const ka = sort.keyFn(a)
      const kb = sort.keyFn(b)
      if (ka < kb) return -1
      if (ka > kb) return 1
      // Tiebreak: session_id for determinism.
      return (a.session_id || '').localeCompare(b.session_id || '')
    })
  }, [sessions, sortKey])

  // Auto-expand the top session on first load for discoverability.
  useMemo(() => {
    if (sorted && sorted.length && expanded.size === 0) {
      setExpanded(new Set([`${sorted[0].source}:${sorted[0].session_id}`]))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sorted])

  if (!sorted) return (
    <div style={{
      padding: 12, fontFamily: 'var(--font-mono)',
      fontSize: 12, color: 'var(--text-secondary)',
    }}>
      Loading sessions…
    </div>
  )

  const setSort = k => {
    setSortKey(k)
    localStorage.setItem('ab-session-sort', k)
  }

  const toggle = key => {
    const next = new Set(expanded)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    setExpanded(next)
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0,
    }}>
      {/* Sort toggle */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '6px 10px',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: 'var(--text-disabled)',
      }}>
        <span>sort</span>
        {Object.keys(SORTS).map(k => {
          const active = sortKey === k
          return (
            <button
              key={k}
              onClick={() => setSort(k)}
              style={{
                padding: '2px 6px',
                background: active ? 'var(--bg-elevated)' : 'transparent',
                border: `1px solid ${active ? 'var(--color-border-hover)' : 'var(--color-border)'}`,
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                borderRadius: 3,
                cursor: 'pointer',
                fontSize: 10,
                fontFamily: 'inherit',
              }}
            >
              {SORTS[k].label}
            </button>
          )
        })}
      </div>
      <div style={{ overflowY: 'auto', flex: 1, minHeight: 0, fontFamily: 'var(--font-mono)' }}>
        {sorted.map(s => {
          const key = `${s.source}:${s.session_id}`
          return (
            <SessionRow
              key={key}
              s={s}
              expanded={expanded.has(key)}
              onToggle={() => toggle(key)}
              onOpenSession={onOpenSession}
              onOpenFile={onOpenFile}
              selectedPath={selectedPath}
              minFileLines={MIN_FILE_LINES_DEFAULT}
              showAll={showAll}
            />
          )
        })}
      </div>
      <div
        onClick={() => setShowAll(v => !v)}
        style={{
          padding: '6px 10px',
          borderTop: '1px solid var(--color-border)',
          cursor: 'pointer', userSelect: 'none',
          fontSize: 10, color: 'var(--text-disabled)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        {showAll ? '✓ showing all files' : `show all files (incl. <${MIN_FILE_LINES_DEFAULT}-line matches)`}
      </div>
    </div>
  )
}
