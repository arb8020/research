import type { ReactNode } from 'react'
import type { TraceSample } from '../types'

export interface SidePanelPluginProps {
  sample: TraceSample
  runId: string
  selectedTurn: number
  checkedTurns: number[]  // sorted ascending
}

export interface SidePanelPlugin {
  /** Tab label shown in the right panel header */
  label: string
  render(props: SidePanelPluginProps): ReactNode
}

// eval_name -> plugin (checked first)
const byEvalName = new Map<string, SidePanelPlugin>()

// environment_state.env_kind -> plugin (fallback)
const byEnvKind = new Map<string, SidePanelPlugin>()

export function registerByEvalName(evalName: string, plugin: SidePanelPlugin): void {
  byEvalName.set(evalName, plugin)
}

export function registerByEnvKind(envKind: string, plugin: SidePanelPlugin): void {
  byEnvKind.set(envKind, plugin)
}

export function resolvePlugin(
  evalName: string | null | undefined,
  sample: TraceSample | null,
): SidePanelPlugin | null {
  if (evalName) {
    const hit = byEvalName.get(evalName)
    if (hit) return hit
  }

  const envState = sample?.environment_state
  if (envState && typeof envState === 'object' && !Array.isArray(envState)) {
    const envKind = (envState as Record<string, unknown>).env_kind
    if (typeof envKind === 'string') {
      const hit = byEnvKind.get(envKind)
      if (hit) return hit
    }
  }

  return null
}
