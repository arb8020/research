import type { RunListItem, RunReport, TraceSample, LiveRun, WorkspaceData } from './types'

export async function listRuns(): Promise<RunListItem[]> {
  const res = await fetch('/api/traces')
  if (!res.ok) throw new Error(`Failed to list runs: ${res.status}`)
  return res.json()
}

export async function getRunReport(runId: string): Promise<{ report: RunReport; sample_ids: string[] }> {
  const res = await fetch(`/api/trace/${runId}`)
  if (!res.ok) throw new Error(`Failed to get run ${runId}: ${res.status}`)
  const data = await res.json()
  return { report: data.report ?? data, sample_ids: data.sample_ids ?? data.report?.sample_ids ?? [] }
}

export async function getSample(runId: string, sampleId: string): Promise<TraceSample> {
  const res = await fetch(`/api/trace/${runId}/sample/${sampleId}`)
  if (!res.ok) throw new Error(`Failed to get sample ${sampleId}: ${res.status}`)
  return res.json()
}

export async function listActiveRuns(): Promise<LiveRun[]> {
  const res = await fetch('/api/runs')
  if (!res.ok) throw new Error(`Failed to list active runs: ${res.status}`)
  const data = await res.json()
  return data.runs ?? []
}

export async function killRun(runId: string): Promise<void> {
  const res = await fetch(`/api/kill/${runId}`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to kill run ${runId}: ${res.status}`)
}

export function openRunStream(runId: string): EventSource {
  return new EventSource(`/api/stream/${runId}`)
}

export function openWatchStream(runId: string): EventSource {
  return new EventSource(`/api/watch/${runId}`)
}

export interface ResultsDirsResponse {
  current: string
  dirs: Array<{ path: string; label: string; exists: boolean }>
}

export async function getResultsDirs(): Promise<ResultsDirsResponse> {
  const res = await fetch('/api/results-dirs')
  if (!res.ok) throw new Error(`Failed to get results dirs: ${res.status}`)
  return res.json()
}

export async function setResultsDir(path: string): Promise<void> {
  const res = await fetch('/api/set-results-dir', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!res.ok) throw new Error(`Failed to set results dir: ${res.status}`)
}

export async function getWorkspace(runId: string, sampleId: string): Promise<WorkspaceData | null> {
  const res = await fetch(`/api/trace/${runId}/sample/${sampleId}/workspace`)
  if (!res.ok) return null
  return res.json()
}

export function getRunHtmlExportUrl(runId: string): string {
  return `/api/trace/${encodeURIComponent(runId)}/export.html`
}

export function getSampleHtmlExportUrl(runId: string, sampleId: string): string {
  return `/api/trace/${encodeURIComponent(runId)}/sample/${encodeURIComponent(sampleId)}/export.html`
}
