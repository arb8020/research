import { useState } from 'react'

function buildTree(files) {
  const root = {}
  for (const file of files) {
    const parts = file.path.split('/')
    let node = root
    for (let i = 0; i < parts.length - 1; i++) {
      if (!node[parts[i]]) node[parts[i]] = { __dir: true, __children: {} }
      node = node[parts[i]].__children
    }
    node[parts[parts.length - 1]] = { __file: file }
  }
  return root
}

const S = {
  dir: {
    display: 'flex', alignItems: 'center', gap: 4,
    padding: '2px 8px', cursor: 'pointer',
    color: 'var(--text-secondary)', userSelect: 'none',
    fontSize: 12, fontFamily: 'var(--font-mono)',
  },
  file: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '2px 8px', cursor: 'pointer',
    color: 'var(--text-secondary)', userSelect: 'none',
    fontSize: 12, fontFamily: 'var(--font-mono)',
  },
}

function DirNode({ name, children, depth, onSelect }) {
  const [open, setOpen] = useState(true)
  return (
    <>
      <div
        style={{ ...S.dir, paddingLeft: 8 + depth * 12 }}
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-component)'}
        onMouseLeave={e => e.currentTarget.style.background = ''}
      >
        <span style={{ fontSize: 9, color: 'var(--text-disabled)', transform: open ? 'rotate(90deg)' : '', display: 'inline-block', transition: 'transform 100ms', width: 10 }}>▶</span>
        <span style={{ opacity: 0.6 }}>📁</span>
        <span>{name}</span>
      </div>
      {open && <TreeNodes node={children} depth={depth + 1} onSelect={onSelect} />}
    </>
  )
}

function FileNode({ name, file, depth, onSelect }) {
  return (
    <div
      style={{ ...S.file, paddingLeft: 16 + depth * 12 }}
      onClick={() => onSelect(file)}
      onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-component)'; e.currentTarget.style.color = 'var(--text-primary)' }}
      onMouseLeave={e => { e.currentTarget.style.background = ''; e.currentTarget.style.color = 'var(--text-secondary)' }}
    >
      <span style={{ opacity: 0.5, fontSize: 11 }}>📄</span>
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
      <span style={{ display: 'flex', gap: 4, flexShrink: 0, fontSize: 11 }}>
        {file.additions > 0 && <span style={{ color: 'var(--text-green)' }}>+{file.additions}</span>}
        {file.deletions > 0 && <span style={{ color: 'var(--text-red)' }}>-{file.deletions}</span>}
      </span>
    </div>
  )
}

function TreeNodes({ node, depth, onSelect }) {
  const entries = Object.entries(node).sort(([a, av], [b, bv]) => {
    if (av.__dir && !bv.__dir) return -1
    if (!av.__dir && bv.__dir) return 1
    return a.localeCompare(b)
  })
  return entries.map(([name, val]) =>
    val.__dir
      ? <DirNode key={name} name={name} children={val.__children} depth={depth} onSelect={onSelect} />
      : <FileNode key={name} name={name} file={val.__file} depth={depth} onSelect={onSelect} />
  )
}

export function FileTree({ files, onSelect }) {
  return (
    <div style={{
      width: 240, minWidth: 240,
      background: 'var(--bg-wash)',
      borderRight: '1px solid var(--color-border)',
      overflowY: 'auto',
      padding: '8px 0',
    }}>
      <div style={{
        padding: '4px 8px 8px',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--text-disabled)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
      }}>
        {files.length} files changed
      </div>
      <TreeNodes node={buildTree(files)} depth={0} onSelect={onSelect} />
    </div>
  )
}
