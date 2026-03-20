import type { RunListItem, RunReport, RunTags, TraceSample, WorkspaceData } from './types'

export async function listRuns(): Promise<RunListItem[]> {
  const res = await fetch('/api/runs')
  if (!res.ok) throw new Error(`Failed to list runs: ${res.status}`)
  const data = await res.json()
  return data.runs ?? []
}

export async function getRunReport(
  runId: string,
): Promise<{ report: RunReport; sample_ids: string[]; tags: RunTags }> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(`Failed to get run ${runId}: ${res.status}`)
  const data = await res.json()
  return {
    report: data.report ?? data,
    sample_ids: data.sample_ids ?? data.report?.sample_ids ?? [],
    tags: data.tags ?? { user: {}, derived: {} },
  }
}

export async function getSample(runId: string, sampleId: string): Promise<TraceSample> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleId)}`)
  if (!res.ok) throw new Error(`Failed to get sample ${sampleId}: ${res.status}`)
  return res.json()
}

export async function killRun(runId: string): Promise<void> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/kill`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to kill run ${runId}: ${res.status}`)
}

export function openRunEvents(runId: string): EventSource {
  return new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`)
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

export async function setRunTags(
  runId: string,
  tags: Record<string, string | null>,
): Promise<Record<string, string>> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/tags`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags }),
  })
  if (!res.ok) throw new Error(`Failed to set tags for run ${runId}: ${res.status}`)
  const data = await res.json()
  return data.user ?? {}
}

export async function getWorkspace(runId: string, sampleId: string): Promise<WorkspaceData | null> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleId)}/workspace`)
  if (!res.ok) return null
  return res.json()
}

export function getRunHtmlExportUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/export.html`
}

export function getSampleHtmlExportUrl(runId: string, sampleId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleId)}/export.html`
}
