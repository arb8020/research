// Divergent copy of ~/research/dev/pr-bot/ui/src/App.jsx.
// pr-bot = per-commit-range diff viewer. agent-blame = whole-file viewer
// with per-line session attribution. Shared substrate: Shiki highlighting,
// Devin-derived CSS tokens, three-column layout, server scaffolding.
// Expected to consolidate once divergence stabilizes.

import { useEffect, useState } from 'react'
import { BlameView } from './BlameView'
import { ChatPanel } from './ChatPanel'
import { SessionPanel } from './SessionPanel'
import { Sidebar } from './Sidebar'
import { useBlame, useFileList, useRepoInfo } from './useBlame'

export default function App() {
  const { info, error: repoError } = useRepoInfo()
  const { files, error: filesError } = useFileList()
  const [selected, setSelected] = useState(null)
  const [selectedEdit, setSelectedEdit] = useState(null)  // {source, session_id, tool_call_id, timestamp, _nonce}
  const { blame, error: blameError } = useBlame(selected)

  // Incrementing nonce so clicking the SAME edit twice still triggers the
  // scroll effect in SessionPanel. Without this, React bails on an
  // identical object and the panel stays where the user scrolled it.
  const selectEdit = edit => setSelectedEdit({
    ...edit,
    _nonce: (selectedEdit?._nonce ?? 0) + 1,
  })

  // Default-select the top attributed file once the list arrives.
  useEffect(() => {
    if (!selected && files && files.length) {
      const top = [...files].sort((a, b) => b.attributed - a.attributed)
        .find(f => f.attributed > 0)
      if (top) setSelected(top.path)
    }
  }, [files, selected])

  const error = repoError || filesError || blameError
  if (error) return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%',
      fontFamily: 'var(--font-mono)', color: 'var(--text-red)', fontSize: 13,
    }}>
      Error: {error}
    </div>
  )

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'var(--bg-page)',
    }}>
      {/* Topbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '0 16px',
        height: 44, flexShrink: 0,
        background: 'var(--bg-wash)',
        borderBottom: '1px solid var(--color-border)',
        fontFamily: 'var(--font-mono)', fontSize: 13,
      }}>
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
          agent-blame
        </span>
        {info && (
          <>
            <span style={{ color: 'var(--text-secondary)' }}>{info.scope}</span>
            <span style={{ color: 'var(--text-disabled)' }}>·</span>
            <span style={{ color: 'var(--text-secondary)' }}>
              {info.source === 'git_sha' ? info.sha?.slice(0, 8) : 'working tree'}
            </span>
            <span style={{ color: 'var(--text-disabled)' }}>·</span>
            <span style={{ color: 'var(--text-secondary)' }}>
              {info.file_count} files
            </span>
          </>
        )}
      </div>

      {/* Three-column layout */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <Sidebar
          files={files}
          selected={selected}
          onSelectFile={setSelected}
          onOpenSession={s => selectEdit({
            source: s.source,
            session_id: s.session_id,
            // no tool_call_id -> SessionPanel scrolls to end
          })}
        />

        <div style={{
          flex: 1, minWidth: 0, overflowY: 'auto',
          background: 'var(--bg-page)',
        }}>
          {selected
            ? <BlameView
                path={selected}
                blame={blame}
                onSelectEdit={selectEdit}
              />
            : <div style={{
                padding: 24, fontFamily: 'var(--font-mono)',
                fontSize: 13, color: 'var(--text-secondary)',
              }}>
                Select a file on the left.
              </div>
          }
        </div>

        {selectedEdit
          ? <SessionPanel
              session={selectedEdit}
              onClose={() => setSelectedEdit(null)}
            />
          : <ChatPanel />}
      </div>
    </div>
  )
}
