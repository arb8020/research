// Right-panel session transcript view.
//
// Triggered when the user clicks a gutter stripe in BlameView. Shows the
// responsible session's messages, scrolled so the target tool call is
// near the top. Read-only for now — chat wiring comes later.

import { useEffect, useMemo, useRef, useState } from 'react'

function relativeTime(iso) {
  if (!iso) return ''
  try {
    const then = new Date(iso).getTime()
    const dt = (Date.now() - then) / 1000
    if (dt < 60) return `${Math.round(dt)}s ago`
    if (dt < 3600) return `${Math.round(dt / 60)}m ago`
    if (dt < 86400) return `${Math.round(dt / 3600)}h ago`
    return `${Math.round(dt / 86400)}d ago`
  } catch {
    return iso
  }
}

function ToolCallBlock({ tc, highlight }) {
  const input = tc.input
  let body
  if (typeof input === 'string') {
    body = input
  } else if (input == null) {
    body = ''
  } else {
    try {
      body = JSON.stringify(input, null, 2)
    } catch {
      body = String(input)
    }
  }
  return (
    <pre style={{
      margin: 0, padding: 8,
      background: 'var(--bg-diff)',
      borderLeft: `2px solid ${highlight ? 'var(--text-green)' : 'var(--color-border)'}`,
      color: 'var(--text-primary)',
      fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: '16px',
      whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      borderRadius: 3,
      maxHeight: highlight ? 'none' : 300,
      overflow: highlight ? 'visible' : 'auto',
    }}>
      <div style={{
        fontSize: 10, color: 'var(--text-disabled)',
        marginBottom: 4, fontWeight: 500,
      }}>
        {tc.name || 'tool'}  ·  {tc.tool_call_id?.slice(0, 16) || ''}
        {highlight && <span style={{ color: 'var(--text-green)', marginLeft: 8 }}>← this edit</span>}
      </div>
      {body}
    </pre>
  )
}

function Message({ m, targetToolCallId, anchorRef }) {
  const roleColors = {
    user: 'var(--text-primary)',
    assistant: 'var(--text-primary)',
    tool_result: 'var(--text-secondary)',
    system: 'var(--text-disabled)',
    developer: 'var(--text-disabled)',
  }
  const roleBgs = {
    user: 'var(--bg-elevated)',
    assistant: 'transparent',
    tool_result: 'var(--bg-surface)',
    system: 'transparent',
    developer: 'transparent',
  }
  const hasTarget = m.tool_calls?.some(tc => tc.tool_call_id === targetToolCallId)
    || m.tool_result_for === targetToolCallId
  return (
    <div
      ref={hasTarget ? anchorRef : null}
      style={{
        padding: '8px 12px',
        borderBottom: '1px solid var(--color-border)',
        background: roleBgs[m.role] || 'transparent',
        color: roleColors[m.role] || 'var(--text-primary)',
        fontFamily: 'var(--font-sans)', fontSize: 12.5, lineHeight: '18px',
      }}
    >
      <div style={{
        fontSize: 10, color: 'var(--text-disabled)',
        marginBottom: 4, letterSpacing: 0.3,
      }}>
        {m.role}{m.tool_result_for ? ` → ${m.tool_result_for.slice(0, 12)}` : ''}
        {m.timestamp && <span style={{ marginLeft: 8 }}>{relativeTime(m.timestamp)}</span>}
      </div>
      {m.text && (
        <div style={{
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          fontFamily: m.role === 'tool_result' ? 'var(--font-mono)' : 'var(--font-sans)',
          fontSize: m.role === 'tool_result' ? 11.5 : 12.5,
        }}>
          {m.text}
        </div>
      )}
      {m.tool_calls?.length > 0 && (
        <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {m.tool_calls.map(tc => (
            <ToolCallBlock
              key={tc.tool_call_id}
              tc={tc}
              highlight={tc.tool_call_id === targetToolCallId}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function SessionPanel({ session, onClose }) {
  const { source, session_id, tool_call_id } = session || {}
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const anchorRef = useRef(null)
  const endRef = useRef(null)

  useEffect(() => {
    if (!source || !session_id) return
    setData(null); setError(null)
    let cancelled = false
    fetch(`/api/session?source=${encodeURIComponent(source)}&session_id=${encodeURIComponent(session_id)}`)
      .then(r => r.ok ? r.json() : r.json().then(j => Promise.reject(j.error || r.statusText)))
      .then(d => { if (!cancelled) setData(d) })
      .catch(e => { if (!cancelled) setError(String(e)) })
    return () => { cancelled = true }
  }, [source, session_id])

  // Codex's tool_call_ids have synthetic `#N` suffixes (MultiEdit-style expansion);
  // strip them so the anchor match hits the base call_id.
  const strippedTargetId = useMemo(() => {
    if (!tool_call_id) return null
    return tool_call_id.split('#')[0]
  }, [tool_call_id])

  // Scroll the anchor into view once layout has completed. The naive
  // useEffect-on-[data] approach fired before React finished painting
  // ~2000 messages, so the ref was sometimes null and the scroll was
  // a no-op. Double rAF waits for one full paint cycle. If the target
  // id isn't present at all (stale blame data, or transcript missing
  // the call for some reason), we fall through to scrolling to the
  // end — matches the "click session header -> go to end" behavior.
  useEffect(() => {
    if (!data) return
    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (anchorRef.current) {
          anchorRef.current.scrollIntoView({ block: 'start', behavior: 'auto' })
        } else if (endRef.current) {
          endRef.current.scrollIntoView({ block: 'end', behavior: 'auto' })
        }
      })
    })
    return () => cancelAnimationFrame(id)
  }, [data, strippedTargetId])

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: 420, flexShrink: 0,
      background: 'var(--bg-wash)',
      borderLeft: '1px solid var(--color-border)',
      minHeight: 0,
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '10px 12px',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 12,
      }}>
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
          {source} {session_id?.slice(0, 8)}
        </span>
        {data && (
          <span style={{ color: 'var(--text-disabled)' }}>
            {data.messages.length} msgs
          </span>
        )}
        <button
          onClick={onClose}
          style={{
            marginLeft: 'auto',
            padding: '2px 8px',
            fontSize: 11, fontFamily: 'var(--font-mono)',
            background: 'transparent',
            border: '1px solid var(--color-border)',
            color: 'var(--text-secondary)',
            borderRadius: 3, cursor: 'pointer',
          }}
        >
          close
        </button>
      </div>

      <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
        {error && (
          <div style={{
            padding: 16, fontFamily: 'var(--font-mono)', fontSize: 12,
            color: 'var(--text-red)',
          }}>
            error: {error}
          </div>
        )}
        {!error && !data && (
          <div style={{
            padding: 16, fontFamily: 'var(--font-mono)', fontSize: 12,
            color: 'var(--text-secondary)',
          }}>
            loading session…
          </div>
        )}
        {data?.messages.map((m, i) => (
          <Message
            key={i}
            m={m}
            targetToolCallId={strippedTargetId}
            anchorRef={anchorRef}
          />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
