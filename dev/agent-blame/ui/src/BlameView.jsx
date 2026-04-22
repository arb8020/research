// BlameView: whole file viewer with two modes, Code and Blame.
//
// Code mode:  [#] [gutter stripe] [code]         (compact, default)
// Blame mode: [stripe] [session  · time] [#] [code]  (wide, GitHub-style)
//
// In blame mode, consecutive same-session lines collapse into a "run": the
// session annotation is shown once at the top of the run with the colored
// stripe spanning the run's full height.
//
// "Bridge" blanks: a lone unknown line between two neighbors attributed to
// the same session gets visually folded into that run. We do NOT mutate the
// backing data — just render bridged lines with the flanking session's
// stripe so the gutter reads as continuous inside authored blocks. Multi-
// line unknown runs break the bridge (that's probably a real human edit).

import { useMemo, useState } from 'react'
import { useSingleFileHighlight } from './useShiki'

const PALETTE = [
  '#4a9eff', '#00ec7e', '#f35b00', '#b685ff', '#ffb84a',
  '#ff6384', '#4aeaff', '#a3e635', '#f472b6', '#fbbf24',
  '#60a5fa', '#fb923c',
]
function colorFor(source, sessionId) {
  if (!sessionId) return 'transparent'
  let h = 0
  const s = `${source}:${sessionId}`
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return PALETTE[Math.abs(h) % PALETTE.length]
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function relativeTime(iso) {
  if (!iso) return ''
  try {
    const then = new Date(iso).getTime()
    const dt = (Date.now() - then) / 1000
    if (dt < 60) return 'just now'
    if (dt < 3600) return `${Math.round(dt / 60)}m ago`
    if (dt < 86400) return `${Math.round(dt / 3600)}h ago`
    if (dt < 86400 * 30) return `${Math.round(dt / 86400)}d ago`
    if (dt < 86400 * 365) return `${Math.round(dt / 86400 / 30)}mo ago`
    return `${Math.round(dt / 86400 / 365)}y ago`
  } catch { return '' }
}

function _runTitle(edit, commit) {
  if (edit) {
    return `${edit.source} ${edit.session_id}\n${new Date(edit.timestamp).toLocaleString()}`
  }
  if (commit) {
    const parts = [
      `commit ${commit.short_sha}`,
      commit.summary || '',
      `${commit.author_name} <${commit.author_email}>`,
      new Date(commit.timestamp).toLocaleString(),
    ]
    if (commit.agent_marker) parts.push(`(${commit.agent_marker} co-authored)`)
    return parts.filter(Boolean).join('\n')
  }
  return 'unknown'
}


function runKey(line) {
  // Group lines by edit (session attribution) when we have one, else
  // by git-commit sha (the fallback). Lines with neither group under
  // a single 'unknown' key.
  if (line.edit) return `edit:${line.edit.source}:${line.edit.session_id}`
  if (line.commit) return `commit:${line.commit.sha}`
  return 'unknown'
}

// Group consecutive same-key lines into runs. A run spans one agent
// session, one git commit, or one stretch of truly-unknown lines.
function computeRuns(lines) {
  const runs = []
  let cur = null
  for (let i = 0; i < lines.length; i++) {
    const key = runKey(lines[i])
    if (!cur || cur.key !== key) {
      if (cur) runs.push(cur)
      cur = {
        key,
        edit: lines[i].edit,
        commit: lines[i].commit,
        start: i,
        end: i,
      }
    } else {
      cur.end = i
    }
  }
  if (cur) runs.push(cur)
  return runs
}

// --------------------------------------------------------------------------
// Code mode: compact per-line layout with interactive gutter.
// --------------------------------------------------------------------------

function HoverCard({ edit, anchorRect }) {
  if (!edit || !anchorRect) return null
  const ts = new Date(edit.timestamp)
  const top = anchorRect.top + anchorRect.height / 2
  const left = anchorRect.right + 12
  return (
    <div style={{
      position: 'fixed',
      top, left, transform: 'translateY(-50%)',
      zIndex: 100,
      padding: '8px 12px',
      background: 'var(--bg-elevated)',
      border: '1px solid var(--color-border)',
      borderRadius: 4,
      fontFamily: 'var(--font-mono)', fontSize: 11.5,
      color: 'var(--text-primary)',
      boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
      pointerEvents: 'none',
      maxWidth: 360,
      whiteSpace: 'nowrap',
    }}>
      <div style={{ fontWeight: 500 }}>
        {edit.source} {edit.session_id.slice(0, 8)}
      </div>
      <div style={{ color: 'var(--text-secondary)', marginTop: 2 }}>
        {ts.toLocaleString()}
      </div>
      <div style={{ color: 'var(--text-disabled)', marginTop: 4, fontSize: 10.5 }}>
        click to open session
      </div>
    </div>
  )
}

function CodeLine({ line, edit, highlightedHtml, onGutterHover, onGutterLeave, onGutterClick, hovered }) {
  const color = edit ? colorFor(edit.source, edit.session_id) : 'transparent'
  const interactive = !!edit
  return (
    <div className="diff-code" style={{
      display: 'flex', alignItems: 'stretch', minHeight: 20,
      width: '100%', minWidth: 0,
    }}>
      <span style={{
        width: 56, flexShrink: 0, textAlign: 'right',
        paddingRight: 10, paddingLeft: 8,
        color: 'var(--text-disabled)', userSelect: 'none',
        lineHeight: '20px',
      }}>
        {line.n}
      </span>
      <span
        onMouseEnter={interactive ? e => onGutterHover(line, edit, e.currentTarget.getBoundingClientRect()) : undefined}
        onMouseLeave={interactive ? onGutterLeave : undefined}
        onClick={interactive ? () => onGutterClick(edit) : undefined}
        style={{
          width: interactive ? (hovered ? 6 : 3) : 3,
          flexShrink: 0,
          background: color,
          alignSelf: 'stretch',
          marginRight: 8,
          cursor: interactive ? 'pointer' : 'default',
          transition: 'width 80ms',
        }}
      />
      <code
        style={{ flex: 1, paddingRight: 24, minWidth: 0, lineHeight: '20px' }}
        dangerouslySetInnerHTML={{
          __html: highlightedHtml || escHtml(line.text) || '&nbsp;',
        }}
      />
    </div>
  )
}

// --------------------------------------------------------------------------
// Blame mode: runs with inline session annotation column.
// --------------------------------------------------------------------------

const ANNOTATION_WIDTH = 200

function BlameRun({ run, lines, highlighted, onSelectEdit }) {
  const { edit, commit, start, end } = run
  const runLen = end - start + 1

  // Three display modes:
  //   agent-run:   edit present -> colored stripe, session_id, click opens transcript
  //   commit-run:  edit null, commit present -> gray stripe, short_sha, author
  //   unknown:     neither -> no stripe, "unknown"
  let stripeColor
  let clickable = false
  if (edit) {
    stripeColor = colorFor(edit.source, edit.session_id)
    clickable = true
  } else if (commit) {
    // Slightly distinctive neutral gray with a hint of warmth if the
    // commit carries an agent marker (i.e. the commit was co-authored
    // with an agent but we don't have that specific session).
    stripeColor = commit.agent_marker ? '#8a7a5a' : '#5a5a5a'
  } else {
    stripeColor = 'transparent'
  }

  return (
    <div style={{
      display: 'flex',
      borderTop: '1px solid var(--color-border)',
    }}>
      <div
        onClick={clickable ? () => onSelectEdit(edit) : undefined}
        style={{
          width: ANNOTATION_WIDTH, flexShrink: 0,
          display: 'flex', alignItems: 'flex-start',
          background: 'var(--bg-wash)',
          borderLeft: `3px solid ${stripeColor}`,
          padding: '2px 10px 2px 8px',
          fontFamily: 'var(--font-mono)', fontSize: 11,
          color: edit ? 'var(--text-secondary)' : 'var(--text-disabled)',
          cursor: clickable ? 'pointer' : 'default',
          overflow: 'hidden',
        }}
        onMouseEnter={clickable ? e => e.currentTarget.style.background = 'var(--bg-elevated)' : undefined}
        onMouseLeave={clickable ? e => e.currentTarget.style.background = 'var(--bg-wash)' : undefined}
        title={_runTitle(edit, commit)}
      >
        {edit ? (
          <div style={{ lineHeight: '20px', overflow: 'hidden', width: '100%' }}>
            <div style={{
              color: 'var(--text-primary)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {edit.source} <span style={{ color: 'var(--text-secondary)' }}>{edit.session_id.slice(0, 8)}</span>
            </div>
            <div style={{
              color: 'var(--text-disabled)', fontSize: 10,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {relativeTime(edit.timestamp)}
              {runLen > 1 && <span style={{ marginLeft: 6 }}>· {runLen} lines</span>}
            </div>
          </div>
        ) : commit ? (
          <div style={{ lineHeight: '20px', overflow: 'hidden', width: '100%' }}>
            <div style={{
              color: 'var(--text-secondary)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              display: 'flex', alignItems: 'center', gap: 6,
            }}>
              <span style={{ color: 'var(--text-disabled)' }}>git</span>
              <span>{commit.short_sha}</span>
              {commit.agent_marker && (
                <span style={{
                  fontSize: 9, padding: '0 4px',
                  background: '#3a3420', color: '#ffcf7a',
                  borderRadius: 2,
                }} title={`commit co-authored with ${commit.agent_marker}`}>
                  {commit.agent_marker === 'claude_code' ? 'cc' :
                   commit.agent_marker === 'codex' ? 'cx' :
                   commit.agent_marker === 'copilot' ? 'co' : 'ag'}
                </span>
              )}
            </div>
            <div style={{
              color: 'var(--text-disabled)', fontSize: 10,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {commit.author_name} · {relativeTime(commit.timestamp)}
              {runLen > 1 && <span style={{ marginLeft: 6 }}>· {runLen} lines</span>}
            </div>
          </div>
        ) : (
          <div style={{ lineHeight: '20px', color: 'var(--text-disabled)' }}>
            unknown{runLen > 1 ? ` · ${runLen} lines` : ''}
          </div>
        )}
      </div>

      {/* Lines in this run */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {Array.from({ length: runLen }, (_, off) => {
          const i = start + off
          const line = lines[i]
          return (
            <div key={line.n} className="diff-code" style={{
              display: 'flex', alignItems: 'stretch', minHeight: 20,
              width: '100%', minWidth: 0,
            }}>
              <span style={{
                width: 56, flexShrink: 0, textAlign: 'right',
                paddingRight: 10, paddingLeft: 8,
                color: 'var(--text-disabled)', userSelect: 'none',
                lineHeight: '20px',
              }}>
                {line.n}
              </span>
              <code
                style={{ flex: 1, paddingRight: 24, minWidth: 0, lineHeight: '20px' }}
                dangerouslySetInnerHTML={{
                  __html: highlighted[i] || escHtml(line.text) || '&nbsp;',
                }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Top-level
// --------------------------------------------------------------------------

export function BlameView({ path, blame, onSelectEdit }) {
  const content = useMemo(
    () => (blame?.lines ?? []).map(l => l.text).join('\n'),
    [blame]
  )
  const highlighted = useSingleFileHighlight({ path, content })
  const [hover, setHover] = useState(null)
  const [mode, setMode] = useState(() => localStorage.getItem('ab-view-mode') || 'code')

  const setModePersisted = m => {
    setMode(m)
    localStorage.setItem('ab-view-mode', m)
  }

  const runs = useMemo(
    () => blame ? computeRuns(blame.lines) : [],
    [blame]
  )

  if (!blame) return (
    <div style={{
      padding: 24, fontFamily: 'var(--font-mono)',
      fontSize: 13, color: 'var(--text-secondary)',
    }}>
      Loading…
    </div>
  )

  const parts = path.split('/')
  const filename = parts.pop()
  const dir = parts.join('/')
  const total = blame.lines.length
  const pct = total ? Math.round(100 * blame.attributed / total) : 0

  return (
    <div style={{
      border: '1px solid var(--color-border)',
      borderRadius: 6, margin: 12, overflow: 'hidden',
      background: 'var(--bg-diff)',
    }}>
      {/* File header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '7px 12px',
        background: 'var(--bg-elevated)',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 13,
      }}>
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{filename}</span>
        {dir && <span style={{ color: 'var(--text-secondary)' }}>{dir}</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Mode toggle — GitHub style */}
          <div style={{
            display: 'flex', border: '1px solid var(--color-border)', borderRadius: 4,
            overflow: 'hidden',
          }}>
            {['code', 'blame'].map(m => (
              <button
                key={m}
                onClick={() => setModePersisted(m)}
                style={{
                  padding: '3px 10px',
                  fontSize: 11, fontFamily: 'var(--font-mono)',
                  background: mode === m ? 'var(--bg-component)' : 'transparent',
                  color: mode === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                {m[0].toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
          <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>
            {blame.attributed}/{total}  ({pct}%)
          </span>
          <span style={{ color: 'var(--text-disabled)', fontSize: 12 }}>
            {blame.unknown} unknown
          </span>
        </div>
      </div>

      {/* Body */}
      <div style={{ overflowX: 'auto' }}>
        {mode === 'code' && blame.lines.map((line, i) => (
          <CodeLine
            key={line.n}
            line={line}
            edit={line.edit}
            highlightedHtml={highlighted[i]}
            hovered={hover?.lineN === line.n}
            onGutterHover={(line, edit, rect) => setHover({ lineN: line.n, edit, rect })}
            onGutterLeave={() => setHover(null)}
            onGutterClick={edit => onSelectEdit && onSelectEdit(edit)}
          />
        ))}
        {mode === 'blame' && runs.map(run => (
          <BlameRun
            key={`${run.start}-${run.end}`}
            run={run}
            lines={blame.lines}
            highlighted={highlighted}
            onSelectEdit={edit => onSelectEdit && onSelectEdit(edit)}
          />
        ))}
      </div>
      {mode === 'code' && <HoverCard edit={hover?.edit} anchorRect={hover?.rect} />}
    </div>
  )
}
