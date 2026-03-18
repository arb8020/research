import { useMemo } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ToolCall } from './ToolCall'
import { ToolResult } from './ToolResult'
import { CodeBlock } from './CodeBlock'
import type { SampleCompletion } from '../types'

interface MessageProps {
  message: {
    role: 'user' | 'assistant' | 'tool'
    content: unknown
    completion?: SampleCompletion
  }
  isExpanded: boolean
  onToggle: () => void
}

function TextWithCodeBlocks({ text }: { text: string }) {
  const parts: React.ReactNode[] = []
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  let key = 0

  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      const before = text.slice(lastIndex, match.index)
      if (before.trim()) {
        parts.push(
          <div key={`t-${key++}`} className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{before}</ReactMarkdown>
          </div>
        )
      }
    }
    parts.push(<CodeBlock key={`c-${key++}`} code={match[2].trim()} language={match[1] || 'text'} />)
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    const remaining = text.slice(lastIndex)
    if (remaining.trim()) {
      parts.push(
        <div key={`t-${key++}`} className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{remaining}</ReactMarkdown>
        </div>
      )
    }
  }

  return parts.length > 0 ? <>{parts}</> : (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  )
}

export function MessageComponent({ message, isExpanded, onToggle }: MessageProps) {
  const { role, completion } = message

  const formattedContent = useMemo(() => {
    // Tool call / tool result objects
    if (role === 'tool' && typeof message.content === 'object' && message.content !== null) {
      const tc = message.content as { type?: string; name?: string; args?: unknown; id?: string; result?: string }
      if (tc.type === 'toolCall' && tc.name) {
        return <ToolCall name={tc.name} args={tc.args ?? {}} id={tc.id} />
      }
      if (tc.type === 'toolResult' && tc.result !== undefined) {
        return <ToolResult result={tc.result} />
      }
    }

    // Plain string
    if (typeof message.content === 'string') {
      const content = message.content.trim()
      if (!content || content.includes('<|tool_calls_section_end|>') || content.includes('<|tool_call_begin|>')) {
        return null
      }
      return <TextWithCodeBlocks text={message.content} />
    }

    // Array of content blocks — render in original order (text inline, toolCalls as chips)
    if (Array.isArray(message.content)) {
      const blocks = message.content
        .map((block, i) => {
          if (typeof block !== 'object' || block === null) return null
          const b = block as { type?: string; text?: string; name?: string; id?: string; arguments?: unknown; input?: unknown }

          if (b.type === 'text' && typeof b.text === 'string') {
            const text = b.text.trim()
            if (!text || text === '<|tool_calls_section_end|>' || text.includes('<|tool_call_begin|>')) return null
            return (
              <div key={i} className="markdown-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{b.text}</ReactMarkdown>
              </div>
            )
          }

          if (b.type === 'toolCall' || b.type === 'tool_use') {
            return (
              <div
                key={i}
                className="my-1 px-2 py-1 rounded"
                style={{ background: 'var(--color-dark-elevated)', border: '1px solid var(--color-dark-border)' }}
              >
                <ToolCall name={b.name ?? 'unknown'} args={b.arguments ?? b.input ?? {}} id={b.id} />
              </div>
            )
          }

          return null
        })
        .filter(Boolean)
      return blocks.length > 0 ? <>{blocks}</> : null
    }

    return null
  }, [message.content, role])

  const roleLabel = role === 'assistant' ? 'Assistant' : role === 'user' ? 'User' : 'Tool'

  const tokenInfo = completion?.usage && role === 'assistant' &&
    (completion.usage.input_tokens + completion.usage.output_tokens > 0) && (
    <div
      className="text-[10px] mt-1 pt-1"
      style={{
        borderTop: '1px solid var(--color-dark-border)',
        color: 'var(--color-dark-text-muted)',
      }}
    >
      {completion.usage.input_tokens} in / {completion.usage.output_tokens} out
      {(completion.usage.cache_read_tokens ?? 0) > 0 && ` · ${completion.usage.cache_read_tokens} cache`}
    </div>
  )

  return (
    <div className="overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-2 py-1 transition-opacity hover:opacity-80 rounded"
        style={{
          background: 'var(--color-dark-elevated)',
          color: 'var(--color-dark-text)',
        }}
      >
        <span className="text-xs font-medium">{roleLabel}</span>
        {isExpanded
          ? <ChevronUp className="h-3 w-3 opacity-60" />
          : <ChevronDown className="h-3 w-3 opacity-60" />
        }
      </button>
      {isExpanded && (
        <div
          className="px-2 py-1.5 overflow-y-auto"
          style={{ maxHeight: '600px' }}
        >
          {formattedContent ? (
            <div className="text-sm">{formattedContent}</div>
          ) : (
            <div className="text-xs italic" style={{ color: 'var(--color-dark-text-muted)' }}>
              (No content)
            </div>
          )}
          {tokenInfo}
        </div>
      )}
    </div>
  )
}
