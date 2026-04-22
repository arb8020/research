// BlameView: whole file with per-line blame gutter.
//
// Replaces pr-bot's DiffView. Layout per line (left -> right):
//
//    [line #]  [session chip]  [code (highlighted)]
//
// The session chip is a colored square per (source, session_id) with the
// 8-char session prefix on hover. `unknown` lines get no chip.
//
// Styling priorities: monospace everything, Geist Mono, tight 20px rows to
// match Devin's diff cadence, borderless within the file body. File header
// reuses pr-bot's chevron + filename + dir layout (Devin-exact) with
// per-file attribution counts replacing the +/- counts.

import { useMemo, useState } from 'react'
import { useSingleFileHighlight } from './useShiki'

// Deterministic color per session. 12-slot palette picked to be distinct
// on Devin's dark background. Color comes from a small hash of source+id.
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

function Gutter({ edit, hovered, onHover, onLeave, onClick }) {
  const color = edit ? colorFor(edit.source, edit.session_id) : 'transparent'
  const interactive = !!edit
  return (
    <span
      onMouseEnter={interactive ? onHover : undefined}
      onMouseLeave={interactive ? onLeave : undefined}
      onClick={interactive ? onClick : undefined}
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
  )
}

function HoverCard({ edit, anchorRect }) {
  if (!edit || !anchorRect) return null
  const ts = new Date(edit.timestamp)
  // Anchor to the right of the gutter stripe, vertically centered on the row.
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
      pointerEvents: 'none', // don't steal hover from the line
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

function Line({ line, highlightedHtml, onGutterHover, onGutterLeave, onGutterClick, hoveredLine }) {
  return (
    <div
      className="diff-code"
      style={{
        display: 'flex', alignItems: 'stretch', minHeight: 20,
        width: '100%', minWidth: 0,
      }}
    >
      <span style={{
        width: 56, flexShrink: 0, textAlign: 'right',
        paddingRight: 10, paddingLeft: 8,
        color: 'var(--text-disabled)', userSelect: 'none',
        lineHeight: '20px',
      }}>
        {line.n}
      </span>
      <Gutter
        edit={line.edit}
        hovered={hoveredLine === line.n}
        onHover={e => onGutterHover(line, e.currentTarget.getBoundingClientRect())}
        onLeave={onGutterLeave}
        onClick={() => onGutterClick(line)}
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

export function BlameView({ path, blame, onSelectEdit }) {
  const content = useMemo(
    () => (blame?.lines ?? []).map(l => l.text).join('\n'),
    [blame]
  )
  const highlighted = useSingleFileHighlight({ path, content })
  const [hover, setHover] = useState(null)  // {line, rect} | null

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
      {/* File header — mirrors pr-bot's but with attribution counts */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '7px 12px',
        background: 'var(--bg-elevated)',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 13,
      }}>
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{filename}</span>
        {dir && <span style={{ color: 'var(--text-secondary)' }}>{dir}</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, fontSize: 12 }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            {blame.attributed}/{total}  ({pct}%)
          </span>
          <span style={{ color: 'var(--text-disabled)' }}>
            {blame.unknown} unknown
          </span>
        </div>
      </div>

      {/* Body */}
      <div style={{ overflowX: 'auto' }}>
        {blame.lines.map((line, i) => (
          <Line
            key={line.n}
            line={line}
            highlightedHtml={highlighted[i]}
            hoveredLine={hover?.line.n}
            onGutterHover={(line, rect) => setHover({ line, rect })}
            onGutterLeave={() => setHover(null)}
            onGutterClick={line => onSelectEdit && line.edit && onSelectEdit(line.edit)}
          />
        ))}
      </div>
      <HoverCard edit={hover?.line.edit} anchorRect={hover?.rect} />
    </div>
  )
}
