// FileList: minimal left sidebar, sorted by attributed-line count.
//
// Placeholder. The user explicitly deprioritized left-sidebar design for
// now. When we do iterate on this, copy Devin's file-tree pattern from the
// HAR's captured JSON shape (sections[]: {title, text, changes}).

export function FileList({ files, selected, onSelect }) {
  if (!files) return (
    <div style={{
      width: 280, padding: 12, fontFamily: 'var(--font-mono)',
      fontSize: 12, color: 'var(--text-secondary)',
      borderRight: '1px solid var(--color-border)',
      background: 'var(--bg-wash)',
    }}>
      Loading files…
    </div>
  )

  const sorted = [...files].sort((a, b) => b.attributed - a.attributed)
  const withAttr = sorted.filter(f => f.attributed > 0)

  return (
    <div style={{
      width: 280, flexShrink: 0, overflowY: 'auto',
      borderRight: '1px solid var(--color-border)',
      background: 'var(--bg-wash)',
      fontFamily: 'var(--font-mono)', fontSize: 12,
    }}>
      <div style={{
        padding: '10px 12px',
        color: 'var(--text-disabled)',
        borderBottom: '1px solid var(--color-border)',
      }}>
        {withAttr.length} files with attribution · {files.length} total
      </div>
      {withAttr.map(f => {
        const selectedThis = selected === f.path
        return (
          <div
            key={f.path}
            onClick={() => onSelect(f.path)}
            style={{
              padding: '6px 12px',
              cursor: 'pointer',
              background: selectedThis ? 'var(--bg-elevated)' : 'transparent',
              color: selectedThis ? 'var(--text-primary)' : 'var(--text-secondary)',
              borderLeft: selectedThis ? '2px solid var(--text-green)' : '2px solid transparent',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}
            onMouseEnter={e => { if (!selectedThis) e.currentTarget.style.background = 'var(--bg-elevated)' }}
            onMouseLeave={e => { if (!selectedThis) e.currentTarget.style.background = 'transparent' }}
            title={`${f.attributed}/${f.total_lines} attributed`}
          >
            <div style={{
              overflow: 'hidden', textOverflow: 'ellipsis',
            }}>{f.path}</div>
            <div style={{
              fontSize: 10, color: 'var(--text-disabled)', marginTop: 2,
            }}>
              {f.attributed}/{f.total_lines}
              {f.top_session && ` · ${f.top_session.source.slice(0,2)} ${f.top_session.session_id.slice(0, 6)}`}
            </div>
          </div>
        )
      })}
    </div>
  )
}
