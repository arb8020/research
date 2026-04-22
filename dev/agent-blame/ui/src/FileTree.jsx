// Nested file tree for the left sidebar.
//
// Divergent from pr-bot's FileTree.jsx (which was a flat-ish file list).
// Builds a directory tree from the flat file-summary list, renders with
// expand/collapse. Each leaf shows filename + attribution count.

import { useEffect, useMemo, useState } from 'react'

function buildTree(files) {
  const root = { name: '', path: '', children: {}, file: null }
  for (const f of files) {
    const parts = f.path.split('/')
    let cur = root
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const isLeaf = i === parts.length - 1
      if (!cur.children[part]) {
        cur.children[part] = {
          name: part,
          path: parts.slice(0, i + 1).join('/'),
          children: {},
          file: null,
        }
      }
      cur = cur.children[part]
      if (isLeaf) cur.file = f
    }
  }
  function sortNode(node) {
    const kids = Object.values(node.children)
    kids.forEach(sortNode)
    kids.sort((a, b) => {
      const aDir = Object.keys(a.children).length > 0
      const bDir = Object.keys(b.children).length > 0
      if (aDir !== bDir) return aDir ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    node._sorted = kids
  }
  sortNode(root)
  return root
}

function hasAttributed(node) {
  if (node.file) return node.file.attributed > 0
  for (const kid of Object.values(node.children)) {
    if (hasAttributed(kid)) return true
  }
  return false
}

function TreeNode({ node, depth, expanded, setExpanded, onSelect, selectedPath }) {
  const isLeaf = !!node.file
  const isOpen = expanded.has(node.path)
  const isSelected = isLeaf && selectedPath === node.path

  const toggle = () => {
    const next = new Set(expanded)
    if (next.has(node.path)) next.delete(node.path)
    else next.add(node.path)
    setExpanded(next)
  }
  const onClick = () => {
    if (isLeaf) onSelect(node.path)
    else toggle()
  }
  const pct = isLeaf && node.file.total_lines
    ? Math.round(100 * node.file.attributed / node.file.total_lines)
    : null

  return (
    <div>
      <div
        onClick={onClick}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '3px 8px',
          paddingLeft: 8 + depth * 12,
          cursor: 'pointer',
          background: isSelected ? 'var(--bg-elevated)' : 'transparent',
          borderLeft: isSelected ? '2px solid var(--text-green)' : '2px solid transparent',
          whiteSpace: 'nowrap',
        }}
        onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'var(--bg-elevated)' }}
        onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent' }}
      >
        {!isLeaf && (
          <span style={{
            fontSize: 8, color: 'var(--text-disabled)',
            width: 8, textAlign: 'center',
            transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 80ms',
          }}>▶</span>
        )}
        {isLeaf && <span style={{ width: 8 }} />}
        <span style={{
          overflow: 'hidden', textOverflow: 'ellipsis', flex: 1,
          color: isLeaf
            ? (isSelected ? 'var(--text-primary)' : 'var(--text-secondary)')
            : 'var(--text-primary)',
        }}>
          {node.name}
        </span>
        {isLeaf && (
          <span style={{ fontSize: 10, color: 'var(--text-disabled)' }}>
            {node.file.attributed}/{node.file.total_lines}
            {pct !== null && <span> · {pct}%</span>}
          </span>
        )}
      </div>
      {!isLeaf && isOpen && node._sorted.map(child => (
        <TreeNode
          key={child.path}
          node={child}
          depth={depth + 1}
          expanded={expanded}
          setExpanded={setExpanded}
          onSelect={onSelect}
          selectedPath={selectedPath}
        />
      ))}
    </div>
  )
}

export function FileTree({ files, selected, onSelect }) {
  const tree = useMemo(() => (files ? buildTree(files) : null), [files])
  const [expanded, setExpanded] = useState(new Set())

  useEffect(() => {
    if (!tree) return
    const next = new Set(expanded)
    // Auto-expand top-level directories that actually have attribution
    // under them. Avoids a wall of empty-looking top dirs.
    for (const kid of tree._sorted) {
      if (Object.keys(kid.children).length > 0 && hasAttributed(kid)) {
        next.add(kid.path)
      }
    }
    // Auto-expand ancestors of the currently-selected file.
    if (selected) {
      const parts = selected.split('/')
      for (let i = 1; i < parts.length; i++) {
        next.add(parts.slice(0, i).join('/'))
      }
    }
    setExpanded(next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree, selected])

  if (!tree) return (
    <div style={{
      padding: 12, fontFamily: 'var(--font-mono)',
      fontSize: 12, color: 'var(--text-secondary)',
    }}>
      Loading files…
    </div>
  )

  return (
    <div style={{
      overflowY: 'auto', flex: 1, minHeight: 0,
      fontFamily: 'var(--font-mono)', fontSize: 12,
    }}>
      {tree._sorted.map(child => (
        <TreeNode
          key={child.path}
          node={child}
          depth={0}
          expanded={expanded}
          setExpanded={setExpanded}
          onSelect={onSelect}
          selectedPath={selected}
        />
      ))}
    </div>
  )
}
