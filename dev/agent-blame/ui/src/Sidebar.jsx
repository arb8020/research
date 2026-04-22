// Left sidebar with a two-tab toggle: Files (tree) / Sessions (grouped).

import { useEffect, useState } from 'react'
import { FileTree } from './FileTree'
import { SessionList } from './SessionList'

function useSessions() {
  const [sessions, setSessions] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    fetch('/api/sessions')
      .then(r => r.json())
      .then(d => setSessions(d.sessions))
      .catch(e => setError(e.message))
  }, [])
  return { sessions, error }
}

export function Sidebar({ files, selected, onSelectFile, onOpenSession }) {
  const [tab, setTab] = useState(() => localStorage.getItem('ab-sidebar-tab') || 'files')
  const { sessions } = useSessions()

  const setTabPersisted = t => {
    setTab(t)
    localStorage.setItem('ab-sidebar-tab', t)
  }

  return (
    <div style={{
      width: 320, flexShrink: 0,
      display: 'flex', flexDirection: 'column',
      borderRight: '1px solid var(--color-border)',
      background: 'var(--bg-wash)',
      minHeight: 0,
    }}>
      {/* Tab header */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 11.5,
      }}>
        {['files', 'sessions'].map(t => {
          const active = tab === t
          return (
            <button
              key={t}
              onClick={() => setTabPersisted(t)}
              style={{
                flex: 1,
                padding: '8px 10px',
                background: active ? 'var(--bg-elevated)' : 'transparent',
                border: 'none',
                borderBottom: active ? '2px solid var(--text-green)' : '2px solid transparent',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontFamily: 'inherit', fontSize: 'inherit',
              }}
            >
              {t}
            </button>
          )
        })}
      </div>

      {tab === 'files' && (
        <FileTree files={files} selected={selected} onSelect={onSelectFile} />
      )}
      {tab === 'sessions' && (
        <SessionList
          sessions={sessions}
          selectedPath={selected}
          onOpenFile={onSelectFile}
          onOpenSession={onOpenSession}
        />
      )}
    </div>
  )
}
