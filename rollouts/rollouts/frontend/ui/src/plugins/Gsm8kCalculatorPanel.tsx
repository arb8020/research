import { useMemo } from 'react'
import type { SidePanelPluginProps } from './registry'

interface CalcStep {
  turn: number
  op: string
  args: Record<string, unknown>
  result: number | null
  toolResponse: string
}

function parseCurrentValue(text: string): number | null {
  const m = text.match(/Current value:\s*([-\d.]+)/i)
  return m ? parseFloat(m[1]) : null
}

function useCalcSteps(messages: Array<{ role: string; content: unknown }>): CalcStep[] {
  return useMemo(() => {
    const steps: CalcStep[] = []
    let turn = 0
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i]
      if (msg.role === 'assistant') {
        const blocks = Array.isArray(msg.content) ? msg.content : []
        for (const block of blocks) {
          if (block && typeof block === 'object' && (block as Record<string, unknown>).type === 'toolCall') {
            const b = block as Record<string, unknown>
            const toolMsg = messages[i + 1]
            const toolText = toolMsg?.role === 'tool' && typeof toolMsg.content === 'string' ? toolMsg.content : ''
            steps.push({
              turn,
              op: typeof b.name === 'string' ? b.name : 'unknown',
              args: typeof b.arguments === 'object' && b.arguments !== null ? (b.arguments as Record<string, unknown>) : {},
              result: parseCurrentValue(toolText),
              toolResponse: toolText,
            })
          }
        }
        turn++
      }
    }
    return steps
  }, [messages])
}

/** Value of register at end of turn `t` (last step of that turn), or null if before any steps */
function valueAtTurn(steps: CalcStep[], turn: number): number | null {
  const stepsUpTo = steps.filter(s => s.turn <= turn)
  return stepsUpTo[stepsUpTo.length - 1]?.result ?? null
}

/** Value of register before turn `t` starts (end of turn t-1) */
function valueBefore(steps: CalcStep[], turn: number): number | null {
  if (turn === 0) return null
  return valueAtTurn(steps, turn - 1)
}

function opLabel(step: CalcStep): string {
  const v = step.args.value
  switch (step.op) {
    case 'clear': return 'clear()'
    case 'add': return `+ ${v}`
    case 'subtract': return `− ${v}`
    case 'multiply': return `× ${v}`
    case 'divide': return `÷ ${v}`
    case 'complete_task': return 'complete_task()'
    default: return step.op
  }
}

function opColor(op: string): string {
  switch (op) {
    case 'add': return '#4ade80'
    case 'subtract': return '#f87171'
    case 'multiply': return '#60a5fa'
    case 'divide': return '#c084fc'
    case 'complete_task': return '#fbbf24'
    default: return 'var(--color-dark-text-muted)'
  }
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-wide mb-2 font-medium" style={{ color: 'var(--color-dark-text-muted)' }}>
      {children}
    </div>
  )
}

export function Gsm8kCalculatorPanel({ sample, selectedTurn, checkedTurns }: SidePanelPluginProps) {
  const messages = (sample.trajectory?.messages ?? []) as Array<{ role: string; content: unknown }>
  const steps = useCalcSteps(messages)

  const groundTruth = typeof sample.ground_truth === 'string' ? sample.ground_truth
    : typeof (sample.input as Record<string, unknown> | null)?.answer === 'string'
      ? (sample.input as Record<string, unknown>).answer as string
      : null
  const correct = sample.reward != null && (sample.reward as number) >= 1.0

  // Mirror WorkspacePanel checkedTurns semantics:
  // 0 checked → state at selectedTurn
  // 1 checked → diff: before that turn → after that turn
  // 2+ checked → diff: before min(checked) → after max(checked)
  let mode: 'state' | 'diff'
  let stateAtTurn: number
  let diffFrom: number
  let diffTo: number

  if (checkedTurns.length === 0) {
    mode = 'state'
    stateAtTurn = selectedTurn
  } else if (checkedTurns.length === 1) {
    mode = 'diff'
    diffFrom = checkedTurns[0]
    diffTo = checkedTurns[0]
  } else {
    mode = 'diff'
    diffFrom = checkedTurns[0]
    diffTo = checkedTurns[checkedTurns.length - 1]
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div
        className="flex items-center justify-between px-3 py-2 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--color-dark-border)' }}
      >
        <span className="text-xs font-semibold font-mono" style={{ color: 'var(--color-dark-text-muted)' }}>
          calculator
        </span>
        <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--color-dark-border)', color: 'var(--color-dark-text-muted)' }}>
          {mode}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {mode === 'state' ? (
          <StateView
            steps={steps}
            turn={stateAtTurn!}
            groundTruth={groundTruth}
            correct={correct}
          />
        ) : (
          <DiffView
            steps={steps}
            fromTurn={diffFrom!}
            toTurn={diffTo!}
            groundTruth={groundTruth}
            correct={correct}
          />
        )}
      </div>
    </div>
  )
}

function StateView({ steps, turn, groundTruth, correct }: {
  steps: CalcStep[]
  turn: number
  groundTruth: string | null
  correct: boolean
}) {
  const value = valueAtTurn(steps, turn)
  const stepsAtTurn = steps.filter(s => s.turn <= turn)
  const lastStep = stepsAtTurn[stepsAtTurn.length - 1]
  const isComplete = lastStep?.op === 'complete_task'

  return (
    <div className="space-y-5">
      <div>
        <Label>register</Label>
        <div className="font-mono text-4xl font-bold" style={{ color: value !== null ? 'var(--color-dark-text)' : 'var(--color-dark-text-muted)' }}>
          {value !== null ? value : '—'}
        </div>
      </div>

      {groundTruth && (
        <div>
          <Label>expected</Label>
          <div className="font-mono text-sm" style={{ color: 'var(--color-dark-text-secondary)' }}>{groundTruth}</div>
        </div>
      )}

      {isComplete && (
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded"
          style={{
            background: correct ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
            color: correct ? '#22c55e' : '#ef4444',
          }}
        >
          {correct ? 'correct' : 'incorrect'}
        </span>
      )}

      <div className="text-xs" style={{ color: 'var(--color-dark-text-muted)' }}>
        turn {turn} · step {stepsAtTurn.length} / {steps.length}
      </div>
    </div>
  )
}

function DiffView({ steps, fromTurn, toTurn, groundTruth, correct }: {
  steps: CalcStep[]
  fromTurn: number
  toTurn: number
  groundTruth: string | null
  correct: boolean
}) {
  // State before fromTurn starts → state after toTurn ends
  const before = valueBefore(steps, fromTurn)
  const after = valueAtTurn(steps, toTurn)

  // All steps that happened in the range [fromTurn, toTurn]
  const rangeSteps = steps.filter(s => s.turn >= fromTurn && s.turn <= toTurn)

  const delta = before !== null && after !== null ? after - before : null

  return (
    <div className="space-y-5">
      {/* Before → after */}
      <div>
        <Label>
          {fromTurn === toTurn ? `turn ${fromTurn}` : `turns ${fromTurn} → ${toTurn}`}
        </Label>
        <div className="flex items-baseline gap-3 font-mono">
          <span className="text-2xl" style={{ color: 'var(--color-dark-text-muted)' }}>
            {before !== null ? before : '—'}
          </span>
          <span style={{ color: 'var(--color-dark-text-muted)' }}>→</span>
          <span className="text-2xl font-bold" style={{ color: 'var(--color-dark-text)' }}>
            {after !== null ? after : '—'}
          </span>
          {delta !== null && (
            <span className="text-sm" style={{ color: delta > 0 ? '#4ade80' : delta < 0 ? '#f87171' : 'var(--color-dark-text-muted)' }}>
              ({delta > 0 ? '+' : ''}{delta})
            </span>
          )}
        </div>
      </div>

      {/* Steps in range */}
      {rangeSteps.length > 0 && (
        <div>
          <Label>operations</Label>
          <div className="space-y-1">
            {rangeSteps.map((s, i) => (
              <div key={i} className="flex items-center gap-3 font-mono text-sm">
                <span style={{ color: opColor(s.op), minWidth: 64 }}>{opLabel(s)}</span>
                <span style={{ color: 'var(--color-dark-text-muted)' }}>→ {s.result ?? '?'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {groundTruth && (
        <div>
          <Label>expected</Label>
          <div className="font-mono text-sm" style={{ color: 'var(--color-dark-text-secondary)' }}>{groundTruth}</div>
        </div>
      )}

      {valueAtTurn(steps, toTurn) !== null && steps.filter(s => s.turn <= toTurn).slice(-1)[0]?.op === 'complete_task' && (
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded"
          style={{
            background: correct ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
            color: correct ? '#22c55e' : '#ef4444',
          }}
        >
          {correct ? 'correct' : 'incorrect'}
        </span>
      )}
    </div>
  )
}
