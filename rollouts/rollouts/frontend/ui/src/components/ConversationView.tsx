import { useState, useMemo } from 'react'
import { Eye } from 'lucide-react'
import type { TraceSample } from '../types'

// ─── Styles (ported from rollout-viewer.html) ─────────────────────────────────
const css = `
.rv-msg { border-radius: 2px; margin-bottom: 4px; }

.rv-msg-row {
  display: flex;
  align-items: stretch;
}
.rv-msg-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  cursor: pointer;
  user-select: none;
  transition: background 100ms;
  flex: 1;
  min-width: 0;
}
.rv-msg-header:hover { background: #1c1c1c; }
.rv-msg-controls {
  display: flex;
  align-items: stretch;
  flex-shrink: 0;
  border-left: 1px solid transparent;
}
.rv-msg-pinned .rv-msg-controls { border-left-color: #3b82f6; }
.rv-msg-control {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  flex-shrink: 0;
}
.rv-msg-control:hover { background: #1c1c1c; }
.rv-msg-control-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 2px;
  border: 1px solid #444;
  background: transparent;
  cursor: pointer;
  color: #737373;
  transition: border-color 100ms, color 100ms, background 100ms;
}
.rv-msg-control-button:hover {
  border-color: #666;
  color: #fafafa;
}
.rv-msg-control-button.rv-active {
  border-color: #3b82f6;
  background: rgba(59,130,246,0.18);
  color: #93c5fd;
}

.rv-role-tag {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 2px 6px;
  border-radius: 2px;
  flex-shrink: 0;
}
.rv-role-user      { background: #1c2a1c; color: #22c55e; }
.rv-role-assistant { background: #1e2a3f; color: #93c5fd; }
.rv-role-tool      { background: #1e1e2e; color: #c4b5fd; }
.rv-role-system    { background: #2a1e1e; color: #fca5a5; }

.rv-msg-subtitle {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  color: #636363;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.rv-collapse-icon { font-size: 10px; color: #636363; transition: transform 100ms; flex-shrink: 0; }

.rv-msg-ts {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  color: #636363;
  margin-left: auto;
  flex-shrink: 0;
}

.rv-msg-body {
  padding: 8px 12px 12px 12px;
  border-top: 1px solid #1e1e1e;
}

.rv-msg-text {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 13px;
  color: #fafafa;
  line-height: 1.6;
}
.rv-msg-text.rv-dim { color: #a3a3a3; font-size: 12px; }

.rv-result-text {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  color: #a3a3a3;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
  max-height: 400px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #333 transparent;
}

.rv-text-block {
  background: #111111;
  border: 1px solid #1e1e1e;
  border-radius: 2px;
  margin-top: 6px;
  overflow: hidden;
}
.rv-text-block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  cursor: pointer;
  user-select: none;
  transition: background 100ms;
}
.rv-text-block-header:hover { background: #1c1c1c; }
.rv-text-block-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #a3a3a3;
}
.rv-text-block-preview {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  color: #636363;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.rv-text-block-body {
  padding: 8px 10px;
  border-top: 1px solid #1e1e1e;
}

.rv-tool-call {
  background: #161616;
  border: 1px solid #262626;
  border-radius: 2px;
  margin-top: 6px;
  overflow: hidden;
}
.rv-tool-call-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  cursor: pointer;
  user-select: none;
  transition: background 100ms;
}
.rv-tool-call-header:hover { background: #1c1c1c; }
.rv-tool-name { font-family: "IBM Plex Mono", monospace; font-size: 11px; font-weight: 600; color: #f59e0b; }
.rv-tool-id   { font-family: "IBM Plex Mono", monospace; font-size: 10px; color: #636363; }
.rv-tool-call-body { padding: 8px 10px; border-top: 1px solid #1e1e1e; }

.rv-tool-output {
  background: #0e1a0e;
  border: 1px solid #1a2e1a;
  border-radius: 2px;
  margin-top: 6px;
  overflow: hidden;
}
.rv-tool-output-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  cursor: pointer;
  user-select: none;
  transition: background 100ms;
}
.rv-tool-output-header:hover { background: #1c1c1c; }
.rv-tool-output-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #86efac;
}
.rv-tool-output-body { padding: 8px 10px; border-top: 1px solid #1a2e1a; }

.rv-thinking-block {
  background: #1a1020;
  border: 1px solid #2d1f4e;
  border-radius: 2px;
  margin-top: 6px;
  overflow: hidden;
}
.rv-thinking-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  cursor: pointer;
  user-select: none;
  transition: background 100ms;
}
.rv-thinking-header:hover { background: rgba(168,85,247,0.05); }
.rv-thinking-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #a855f7;
}
.rv-thinking-body {
  padding: 8px 10px;
  border-top: 1px solid #2d1f4e;
  font-size: 12px;
  color: #c4b5fd;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
  font-family: "IBM Plex Mono", monospace;
}

pre.rv-pre {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  background: #0a0a0a;
  border: 1px solid #1e1e1e;
  border-radius: 2px;
  padding: 8px;
  overflow-x: auto;
  white-space: pre;
  color: #a3a3a3;
  line-height: 1.5;
  scrollbar-width: thin;
  scrollbar-color: #333 transparent;
}

/* JSON syntax */
.rv-jk { color: #93c5fd; }
.rv-js { color: #86efac; }
.rv-jn { color: #f9a8d4; }
.rv-jb { color: #fca5a5; }
.rv-jp { color: #636363; }
`

// ─── Parse rollout messages (ported from rollout-viewer.html) ──────────────────

interface ParsedBlock {
  type: 'text' | 'thinking' | 'toolCall' | 'result' | 'image'
  text?: string
  thinking?: string
  encrypted?: boolean
  id?: string
  name?: string
  arguments?: Record<string, unknown>
  // result
  mediaType?: string
  data?: string
}

interface ParsedMessage {
  role: string
  timestamp?: string
  subtitle?: string
  blocks: ParsedBlock[]
}

function splitAssistantBlocks(blocks: ParsedBlock[]): ParsedBlock[][] {
  const segments: ParsedBlock[][] = []
  let current: ParsedBlock[] = []

  const flush = () => {
    if (current.length > 0) {
      segments.push(current)
      current = []
    }
  }

  for (const block of blocks) {
    const resumesAssistantNarration =
      (block.type === 'text' || block.type === 'thinking') &&
      current.some(existing => existing.type === 'toolCall')

    if (resumesAssistantNarration) {
      flush()
    }

    current.push(block)
  }

  flush()
  return segments
}

function parseMessages(messages: TraceSample['trajectory']['messages']): ParsedMessage[] {
  if (!messages) return []

  // Build tool_call_id -> call info map
  const callMap: Record<string, { name: string; arguments: unknown }> = {}
  for (const msg of messages) {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      for (const block of msg.content as Array<Record<string, unknown>>) {
        if (block.type === 'toolCall' && block.id) {
          callMap[block.id as string] = { name: block.name as string, arguments: block.arguments as unknown }
        }
      }
    }
  }

  const parsed: ParsedMessage[] = []
  for (const msg of messages) {
    const role = msg.role
    const content = msg.content

    if (role === 'user' || role === 'system') {
      const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2)
      parsed.push({ role, timestamp: msg.timestamp as string | undefined, blocks: [{ type: 'text', text }] })

    } else if (role === 'tool') {
      const ref = callMap[msg.tool_call_id as string] || {}
      let blocks: ParsedBlock[]
      if (Array.isArray(content)) {
        blocks = (content as Array<Record<string, unknown>>).map(b => {
          if (b.type === 'image') return { type: 'image' as const, mediaType: b.media_type as string || 'image/png', data: b.data as string }
          return { type: 'result' as const, text: typeof b.text === 'string' ? b.text : JSON.stringify(b) }
        })
      } else {
        blocks = [{ type: 'result', text: typeof content === 'string' ? content : JSON.stringify(content, null, 2) }]
      }
      parsed.push({
        role: 'tool',
        timestamp: msg.timestamp as string | undefined,
        subtitle: (ref.name as string) ? `${ref.name} (${msg.tool_call_id})` : msg.tool_call_id as string,
        blocks,
      })

    } else if (role === 'assistant') {
      const rawBlocks = Array.isArray(content) ? content as Array<Record<string, unknown>> : [{ type: 'text', text: content }]
      const normalized: ParsedBlock[] = rawBlocks.map(b => {
        if (b.type === 'thinking') return { type: 'thinking' as const, thinking: b.thinking as string || '', encrypted: !b.thinking }
        if (b.type === 'text') return { type: 'text' as const, text: b.text as string || '' }
        if (b.type === 'toolCall') return { type: 'toolCall' as const, id: b.id as string, name: b.name as string, arguments: b.arguments as Record<string, unknown> }
        return { type: 'text' as const, text: JSON.stringify(b) }
      })
      for (const segment of splitAssistantBlocks(normalized)) {
        parsed.push({ role: 'assistant', timestamp: msg.timestamp as string | undefined, blocks: segment })
      }
    }
  }
  return parsed
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function msgPreview(blocks: ParsedBlock[]): string {
  if (!blocks.length) return ''
  return blocks.map(b => {
    if (b.type === 'text') return (b.text || '').replace(/\s+/g, ' ').trim()
    if (b.type === 'result') return (b.text || '').replace(/\s+/g, ' ').trim()
    if (b.type === 'image') return '[image]'
    if (b.type === 'thinking') {
      const t = (b.thinking || '').replace(/\s+/g, ' ').trim()
      return t ? `[thinking] ${t}` : '[thinking]'
    }
    if (b.type === 'toolCall') {
      const name = b.name || '?'
      const args = b.arguments
      if (!args || Object.keys(args).length === 0) return `${name}()`
      const pairs = Object.entries(args).map(([k, v]) => {
        const s = typeof v === 'string' ? v : JSON.stringify(v)
        return `${k}=${s.length > 40 ? s.slice(0, 40) + '…' : s}`
      })
      return `${name}(${pairs.join(', ')})`
    }
    return ''
  }).filter(Boolean).join('  ·  ')
}

function fmtTs(iso?: string): string {
  if (!iso) return ''
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function highlightJson(obj: unknown): string {
  const raw = JSON.stringify(obj, null, 2)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  return raw.replace(
    /("(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|[[\]{},:])/g,
    m => {
      if (/^".*":$/.test(m))   return `<span class="rv-jk">${m}</span>`
      if (/^"/.test(m))         return `<span class="rv-js">${m}</span>`
      if (/true|false/.test(m)) return `<span class="rv-jb">${m}</span>`
      if (/null/.test(m))       return `<span class="rv-jb">${m}</span>`
      if (/[[\]{},:]/.test(m)) return `<span class="rv-jp">${m}</span>`
      return `<span class="rv-jn">${m}</span>`
    }
  )
}

function toggleSetEntry(current: Set<number>, index: number, isOpen: boolean): Set<number> {
  const next = new Set(current)
  if (isOpen) {
    next.delete(index)
  } else {
    next.add(index)
  }
  return next
}

// ─── Message component ────────────────────────────────────────────────────────

function Message({ msg, isPinned, isSelectedTurn, onTogglePin, onSelectTurn }: { msg: ParsedMessage; isPinned?: boolean; isSelectedTurn?: boolean; onTogglePin?: () => void; onSelectTurn?: () => void }) {
  const [open, setOpen] = useState(false)
  const [openText, setOpenText] = useState<Set<number>>(new Set())
  const [openThinking, setOpenThinking] = useState<Set<number>>(new Set())
  const [openToolCalls, setOpenToolCalls] = useState<Set<number>>(new Set())

  const preview = msg.subtitle || msgPreview(msg.blocks)

  return (
    <div className={isPinned ? 'rv-msg rv-msg-pinned' : 'rv-msg'}>
      <div className="rv-msg-row">
        <div className="rv-msg-header" onClick={() => setOpen(o => !o)}>
          <span className="rv-collapse-icon" style={{ transform: open ? '' : 'rotate(-90deg)' }}>▾</span>
          <span className={`rv-role-tag rv-role-${msg.role}`}>{msg.role}</span>
          {preview && <span className="rv-msg-subtitle">{preview}</span>}
          {msg.timestamp && <span className="rv-msg-ts">{fmtTs(msg.timestamp)}</span>}
        </div>
        {(onSelectTurn || onTogglePin) && (
          <div className="rv-msg-controls">
            {onSelectTurn && (
              <div className="rv-msg-control">
                <button
                  type="button"
                  className={`rv-msg-control-button${isSelectedTurn ? ' rv-active' : ''}`}
                  title="View this turn in the right panel"
                  aria-label="View this turn in the right panel"
                  onClick={e => { e.stopPropagation(); onSelectTurn() }}
                >
                  <Eye size={11} />
                </button>
              </div>
            )}
            {onTogglePin && (
              <div className="rv-msg-control">
                <button
                  type="button"
                  className={`rv-msg-control-button${isPinned ? ' rv-active' : ''}`}
                  title="Add or remove this turn from the comparison interval"
                  aria-label="Add or remove this turn from the comparison interval"
                  onClick={e => { e.stopPropagation(); onTogglePin() }}
                >
                  {isPinned ? (
                    <svg width="7" height="7" viewBox="0 0 7 7" fill="none">
                      <path d="M1 3.5l1.8 1.8 3.2-3.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    <span style={{ fontSize: 10, lineHeight: 1 }}>+</span>
                  )}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {open && (
        <div className="rv-msg-body">
          {msg.blocks.map((block, i) => {
            if (block.type === 'text') {
              // user/system: flat text. assistant: collapsible sub-block
              if (msg.role !== 'assistant') {
                return (
                  <div key={i} className={`rv-msg-text${msg.role === 'system' ? ' rv-dim' : ''}`}>
                    {block.text}
                  </div>
                )
              }
              const isOpen = openText.has(i)
              const preview = (block.text || '').replace(/\s+/g, ' ').trim().slice(0, 100)
              return (
                <div key={i} className="rv-text-block">
                  <div
                    className="rv-text-block-header"
                    onClick={e => {
                      e.stopPropagation()
                      setOpenText(s => toggleSetEntry(s, i, isOpen))
                    }}
                  >
                    <span className="rv-text-block-label">text</span>
                    {!isOpen && <span className="rv-text-block-preview">{preview}</span>}
                    <span className="rv-collapse-icon" style={{ transform: isOpen ? '' : 'rotate(-90deg)' }}>▾</span>
                  </div>
                  {isOpen && (
                    <div className="rv-text-block-body">
                      <div className="rv-msg-text">{block.text}</div>
                    </div>
                  )}
                </div>
              )
            }

            if (block.type === 'result') {
              return (
                <div key={i} className="rv-result-text">
                  {block.text}
                </div>
              )
            }

            if (block.type === 'image') {
              return (
                <img
                  key={i}
                  src={`data:${block.mediaType};base64,${block.data}`}
                  style={{ maxWidth: '100%', height: 'auto', display: 'block', borderRadius: 2, marginTop: 4 }}
                />
              )
            }

            if (block.type === 'thinking') {
              const isOpen = openThinking.has(i)
              return (
                <div key={i} className="rv-thinking-block">
                  <div
                    className="rv-thinking-header"
                    onClick={e => {
                      e.stopPropagation()
                      setOpenThinking(s => toggleSetEntry(s, i, isOpen))
                    }}
                  >
                    <span className="rv-thinking-label">thinking</span>
                    <span className="rv-collapse-icon" style={{ transform: isOpen ? '' : 'rotate(-90deg)' }}>▾</span>
                  </div>
                  {isOpen && (
                    <div className="rv-thinking-body">{block.thinking || '[encrypted]'}</div>
                  )}
                </div>
              )
            }

            if (block.type === 'toolCall') {
              const isOpen = openToolCalls.has(i)
              const hasArgs = block.arguments && Object.keys(block.arguments).length > 0
              return (
                <div key={i} className="rv-tool-call">
                  <div
                    className="rv-tool-call-header"
                    onClick={e => {
                      e.stopPropagation()
                      setOpenToolCalls(s => toggleSetEntry(s, i, isOpen))
                    }}
                  >
                    <span className="rv-tool-name">{block.name || '?'}</span>
                    <span className="rv-tool-id">{block.id || ''}</span>
                    <span className="rv-collapse-icon" style={{ transform: isOpen ? '' : 'rotate(-90deg)' }}>▾</span>
                  </div>
                  {isOpen && (
                    <div className="rv-tool-call-body">
                      {hasArgs && (
                        <pre
                          className="rv-pre"
                          dangerouslySetInnerHTML={{ __html: highlightJson(block.arguments) }}
                        />
                      )}
                    </div>
                  )}
                </div>
              )
            }

            return null
          })}
        </div>
      )}
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

// Inject styles once
let stylesInjected = false
function ensureStyles() {
  if (stylesInjected) return
  stylesInjected = true
  const el = document.createElement('style')
  el.textContent = css
  document.head.appendChild(el)
}

export function ConversationView({
  sample,
  selectedTurn,
  onMessageVisible,
  checkedTurns,
  onToggleTurn,
}: {
  sample: TraceSample
  selectedTurn?: number
  onMessageVisible?: (messageIndex: number) => void
  // TODO: Future improvement — group all messages per turn into collapsible rows
  // and put the checkbox on the turn row rather than individual assistant messages.
  // Currently checkboxes appear only on assistant messages (natural turn boundaries).
  checkedTurns?: Set<number>
  onToggleTurn?: (turn: number) => void
}) {
  ensureStyles()
  const messages = useMemo(() => parseMessages(sample.trajectory.messages), [sample])

  // Track display turns: UI-only projection over canonical assistant messages.
  let turnCounter = -1

  if (!messages.length) {
    return (
      <div style={{ color: '#636363', fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, fontStyle: 'italic' }}>
        No recorded agent turns for this sample
      </div>
    )
  }

  return (
    <div>
      {messages.map((msg, i) => {
        const isAssistant = msg.role === 'assistant'
        if (isAssistant) turnCounter++
        const turn = turnCounter
        const isChecked = isAssistant && checkedTurns?.has(turn)

        return (
          <div key={i}>
            <Message
              msg={msg}
              isPinned={isChecked}
              isSelectedTurn={isAssistant && turn === selectedTurn}
              onSelectTurn={isAssistant && onMessageVisible ? () => onMessageVisible(i) : undefined}
              onTogglePin={isAssistant && onToggleTurn ? () => onToggleTurn(turn) : undefined}
            />
          </div>
        )
      })}
    </div>
  )
}
