// Data-fetch hooks for agent-blame.
//
// Divergent from pr-bot — pr-bot fetches one big /api/diff payload at load
// time. agent-blame fetches per-file /api/blame lazily because a repo can
// have thousands of files and their line counts are much larger than a
// PR diff's. Only the file list is fetched up front.

import { useEffect, useState } from 'react'

export function useRepoInfo() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    fetch('/api/repo').then(r => r.json()).then(setInfo).catch(e => setError(e.message))
  }, [])
  return { info, error }
}

export function useFileList() {
  const [files, setFiles] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    fetch('/api/files')
      .then(r => r.json())
      .then(d => setFiles(d.files))
      .catch(e => setError(e.message))
  }, [])
  return { files, error }
}

export function useBlame(path) {
  const [blame, setBlame] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (!path) { setBlame(null); return }
    setBlame(null); setError(null)
    let cancelled = false
    fetch(`/api/blame?path=${encodeURIComponent(path)}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) setBlame(d) })
      .catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [path])
  return { blame, error }
}
