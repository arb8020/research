import { useEffect, useRef } from 'react'
import { useLiveRun } from '../hooks/useLiveRun'
import { TextWithCodeBlocks } from './Message'
import type { LiveRun, LiveMessage } from '../types'

interface LiveSampleViewerProps {
  run: LiveRun
  sampleId: string
  onBack: () => void
}

function LiveMessageBlock({ msg, turn }: { msg: LiveMessage; turn: number }) {
  return (
    <div
      className="rv-msg"
      style={{
        borderLeft: '2px solid var(--color-neutral-700)',
        paddingLeft: 12,
        marginBottom: 20,
      }}
    >
      <div
        className="text-xs mb-2 font-mono"
        style={{ color: 'var(--color-dark-text-muted)' }}
      >
        turn {turn} · assistant
      </div>
      <div style={{ color: 'var(--color-dark-text)', fontSize: 14, lineHeight: 1.6 }}>
        <TextWithCodeBlocks text={msg.content} />
      </div>
    </div>
  )
}

export function LiveSampleViewer({ run, sampleId, onBack }: LiveSampleViewerProps) {
  const state = useLiveRun(run.run_id, run.status)
  const sample = state.samples.get(sampleId)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom as messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sample?.messages.length])

  const isRunning = sample?.status === 'running' || state.status === 'running'

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: 'var(--color-dark-bg)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 flex-shrink-0"
        style={{
          height: 44,
          borderBottom: '1px solid var(--color-dark-border)',
          background: 'var(--color-dark-surface)',
        }}
      >
        <button
          onClick={onBack}
          className="text-xs hover:opacity-80 transition-opacity font-mono"
          style={{ color: 'var(--color-dark-text-muted)' }}
        >
          ← back
        </button>
        <span className="text-xs font-mono" style={{ color: 'var(--color-dark-text-secondary)' }}>
          {sampleId}
        </span>
        {sample && (
          <span className="text-xs ml-auto font-mono" style={{ color: 'var(--color-dark-text-muted)' }}>
            {sample.status === 'done' && sample.score !== null
              ? `score: ${sample.score.toFixed(3)}`
              : isRunning
              ? `turn ${sample.turn} · running`
              : sample.status}
          </span>
        )}
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px 24px',
        }}
      >
        {!sample || sample.messages.length === 0 ? (
          <div
            className="text-sm font-mono"
            style={{ color: 'var(--color-dark-text-muted)', padding: '40px 0' }}
          >
            {isRunning ? 'waiting for first message...' : 'no messages recorded'}
          </div>
        ) : (
          sample.messages.map((msg, i) => (
            <LiveMessageBlock key={i} msg={msg} turn={msg.turn} />
          ))
        )}

        {/* Running indicator */}
        {isRunning && sample && sample.messages.length > 0 && (
          <div
            className="text-xs font-mono"
            style={{ color: 'var(--color-dark-text-muted)', paddingBottom: 12 }}
          >
            <span style={{ animation: 'pulse-subtle 1.5s ease-in-out infinite', display: 'inline-block' }}>
              ···
            </span>
            {' '}turn {sample.turn}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
