// Divergent copy of pr-bot/ui/src/ChatPanel.jsx.
// pr-bot: general chat with the diff. agent-blame: general chat by default;
// later, click a blamed line -> scope this panel to that session's
// transcript around the responsible tool call. The session prop will
// carry {source, session_id, tool_call_id} for scoped mode.

import { useState, useRef, useEffect } from 'react'

const SUGGESTED = [
  { icon: '🔍', label: 'Who wrote the most?' },
  { icon: '📝', label: 'Summarize this file' },
  { icon: '🏗', label: 'Why was this written?' },
]

export function ChatPanel({ session }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: session
      ? `Scoped to session ${session.session_id.slice(0,8)}. Ask about what this session did or why.`
      : "Browse files on the left to see per-line attribution. Click a line's gutter (coming soon) to scope this chat to the session that wrote it."
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(text) {
    text = (text ?? input).trim()
    if (!text || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: text }])
    setLoading(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, payload }),
      })
      const data = await res.json()
      setMessages(m => [...m, { role: 'assistant', content: data.response }])
    } catch (e) {
      setMessages(m => [...m, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  const showSuggested = messages.length <= 1

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: 280, minWidth: 280,
      background: 'var(--bg-wash)',
      borderLeft: '1px solid var(--color-border)',
    }}>
      {/* Tabs */}
      <div style={{
        display: 'flex', alignItems: 'flex-end', gap: 16,
        padding: '0 12px',
        borderBottom: '1px solid var(--color-border)',
        height: 36, flexShrink: 0,
        fontFamily: 'var(--font-sans)', fontSize: 13,
      }}>
        <span style={{
          color: 'var(--text-primary)',
          borderBottom: '2px solid var(--text-primary)',
          paddingBottom: 6,
          cursor: 'default', userSelect: 'none',
        }}>Chat</span>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-disabled)' }}>
              {msg.role === 'assistant' ? 'pr-bot' : 'you'}
            </span>
            <p style={{
              fontFamily: 'var(--font-sans)', fontSize: 13, lineHeight: 1.55,
              color: msg.role === 'assistant' ? 'var(--text-primary)' : 'var(--text-secondary)',
              margin: 0,
            }}>
              {msg.content}
            </p>
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-disabled)' }}>pr-bot</span>
            <p style={{ fontFamily: 'var(--font-sans)', fontSize: 13, color: 'var(--text-secondary)', margin: 0 }}>Thinking…</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggested prompts */}
      {showSuggested && (
        <div style={{ padding: '0 10px 10px', display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 11, color: 'var(--text-disabled)', marginBottom: 2 }}>
            Suggested prompts
          </span>
          {SUGGESTED.map(s => (
            <button
              key={s.label}
              onClick={() => send(s.label)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '6px 10px',
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: 'var(--bg-elevated)',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-sans)', fontSize: 12,
                cursor: 'pointer', textAlign: 'left',
              }}
              onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-component)'; e.currentTarget.style.color = 'var(--text-primary)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-secondary)' }}
            >
              <span>{s.icon}</span>
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ padding: 8, borderTop: '1px solid var(--color-border)', flexShrink: 0 }}>
        <div style={{
          display: 'flex', alignItems: 'flex-end',
          border: '1px solid var(--color-border)',
          borderRadius: 6,
          background: 'var(--bg-elevated)',
        }}>
          <textarea
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)', fontSize: 13,
              padding: '8px 10px', resize: 'none',
              minHeight: 36, maxHeight: 120,
            }}
            placeholder="Ask anything about this diff…"
            value={input}
            rows={1}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          />
          <button
            onClick={() => send()}
            style={{
              background: 'transparent', border: 'none',
              color: 'var(--text-secondary)', cursor: 'pointer',
              padding: '8px 10px', fontSize: 14, lineHeight: 1,
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-secondary)'}
          >↑</button>
        </div>
      </div>
    </div>
  )
}
