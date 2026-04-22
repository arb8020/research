import { useState } from 'react'
import { useFileHighlight } from './useShiki'

function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function CollapsedRegion({ lineCount, onExpand }) {
  return (
    <div
      onClick={onExpand}
      style={{
        display: 'flex', alignItems: 'center', gap: 16,
        padding: '4px 12px',
        fontFamily: 'var(--font-mono)', fontSize: 12,
        color: '#4a9eff',
        background: '#0d1929',
        borderTop: '1px solid #1a2a40', borderBottom: '1px solid #1a2a40',
        cursor: 'pointer', userSelect: 'none',
      }}
      onMouseEnter={e => e.currentTarget.style.background = '#0f2035'}
      onMouseLeave={e => e.currentTarget.style.background = '#0d1929'}
    >
      <span>↕ {lineCount} lines</span>
      <span style={{ color: '#2a5a8a' }}>·</span>
      <span>↑ All {lineCount} lines</span>
      <span style={{ color: '#2a5a8a' }}>·</span>
      <span>↑ {Math.min(lineCount, 5)} lines</span>
    </div>
  )
}

function UnifiedLine({ lineNum, type, html, fallback }) {
  const numColor = type === 'added' ? 'var(--text-green)' : type === 'removed' ? 'var(--text-red)' : 'var(--text-disabled)'
  const rowClass = type === 'added' ? 'diff-row-added' : type === 'removed' ? 'diff-row-removed' : ''

  return (
    <div className={rowClass} style={{ display: 'flex', width: '100%', minWidth: 0 }}>
      <span className="diff-code" style={{
        width: 48, flexShrink: 0, textAlign: 'right',
        paddingRight: 12, paddingLeft: 8,
        color: numColor, userSelect: 'none',
        tabularNums: true,
      }}>
        {lineNum ?? ''}
      </span>
      <code
        className="diff-code"
        style={{ flex: 1, paddingRight: 24, minWidth: 0 }}
        dangerouslySetInnerHTML={{ __html: html || escHtml(fallback) || '&nbsp;' }}
      />
    </div>
  )
}

function SplitLine({ leftNum, rightNum, leftType, rightType, leftHtml, rightHtml, leftFallback, rightFallback }) {
  const leftNumColor = leftType === 'removed' ? 'var(--text-red)' : 'var(--text-disabled)'
  const rightNumColor = rightType === 'added' ? 'var(--text-green)' : 'var(--text-disabled)'
  const leftClass = leftType === 'removed' ? 'diff-row-removed' : ''
  const rightClass = rightType === 'added' ? 'diff-row-added' : ''

  return (
    <div style={{ display: 'flex', width: '100%', minWidth: 0 }}>
      <div className={leftClass} style={{ display: 'flex', flex: 1, minWidth: 0, borderRight: '1px solid var(--color-border)' }}>
        <span className="diff-code" style={{ width: 48, flexShrink: 0, textAlign: 'right', paddingRight: 12, paddingLeft: 8, color: leftNumColor, userSelect: 'none' }}>
          {leftNum ?? ''}
        </span>
        <code className="diff-code" style={{ flex: 1, paddingRight: 16, minWidth: 0 }}
          dangerouslySetInnerHTML={{ __html: leftHtml || escHtml(leftFallback) || '&nbsp;' }} />
      </div>
      <div className={rightClass} style={{ display: 'flex', flex: 1, minWidth: 0 }}>
        <span className="diff-code" style={{ width: 48, flexShrink: 0, textAlign: 'right', paddingRight: 12, paddingLeft: 8, color: rightNumColor, userSelect: 'none' }}>
          {rightNum ?? ''}
        </span>
        <code className="diff-code" style={{ flex: 1, paddingRight: 16, minWidth: 0 }}
          dangerouslySetInnerHTML={{ __html: rightHtml || escHtml(rightFallback) || '&nbsp;' }} />
      </div>
    </div>
  )
}

function UnifiedDiff({ file, tokens }) {
  const [expanded, setExpanded] = useState({})
  const blocks = file.blocks_to_show
  const items = []

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]
    const prev = blocks[i - 1]

    if (prev) {
      const prevEnd = prev.type === 'line'
        ? prev.line_after
        : (prev.after_lines?.[1] ?? prev.before_lines?.[1] ?? null)
      const thisStart = block.type === 'line'
        ? block.line_after - 1
        : ((block.after_lines?.[0] ?? block.before_lines?.[0] ?? 1) - 1)

      if (prevEnd != null && thisStart != null && thisStart - prevEnd > 1) {
        const key = `gap-${i}`
        const lineCount = thisStart - prevEnd
        if (!expanded[key]) {
          items.push({ type: 'collapsed', key, lineCount, start: prevEnd + 1 })
          items.push({ type: 'block', block })
          continue
        }
        const fileLines = (file.head_content || '').split('\n')
        for (let ln = prevEnd + 1; ln <= thisStart; ln++) {
          items.push({ type: 'expanded-line', lineNum: ln, content: fileLines[ln - 1] ?? '' })
        }
      }
    }
    items.push({ type: 'block', block })
  }

  return (
    <div style={{ width: '100%' }}>
      {items.map((item, idx) => {
        if (item.type === 'collapsed') {
          return <CollapsedRegion key={item.key} lineCount={item.lineCount}
            onExpand={() => setExpanded(e => ({ ...e, [item.key]: true }))} />
        }
        if (item.type === 'expanded-line') {
          return <UnifiedLine key={`exp-${item.lineNum}`} lineNum={item.lineNum} type="context"
            html={tokens.head[item.lineNum - 1]} fallback={item.content} />
        }
        const block = item.block
        if (block.type === 'line') {
          return <UnifiedLine key={idx} lineNum={block.line_after} type="context"
            html={tokens.head[block.line_after - 1]} fallback={block.content} />
        }
        if (block.type === 'hunk') {
          const before = block.before_content ? block.before_content.split('\n') : []
          const after = block.after_content ? block.after_content.split('\n') : []
          return (
            <div key={idx}>
              {before.map((line, i) => {
                const ln = block.before_lines ? block.before_lines[0] + i : null
                return <UnifiedLine key={`r${i}`} lineNum={ln} type="removed" html={tokens.base[ln - 1]} fallback={line} />
              })}
              {after.map((line, i) => {
                const ln = block.after_lines ? block.after_lines[0] + i : null
                return <UnifiedLine key={`a${i}`} lineNum={ln} type="added" html={tokens.head[ln - 1]} fallback={line} />
              })}
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

function SplitDiff({ file, tokens }) {
  return (
    <div style={{ width: '100%' }}>
      {file.blocks_to_show.map((block, bi) => {
        if (block.type === 'line') {
          const html = tokens.head[block.line_after - 1]
          return <SplitLine key={bi}
            leftNum={block.line_before} rightNum={block.line_after}
            leftType="context" rightType="context"
            leftHtml={html} rightHtml={html}
            leftFallback={block.content} rightFallback={block.content} />
        }
        if (block.type === 'hunk') {
          const before = block.before_content ? block.before_content.split('\n') : []
          const after = block.after_content ? block.after_content.split('\n') : []
          const maxLen = Math.max(before.length, after.length)
          return (
            <div key={bi}>
              {Array.from({ length: maxLen }, (_, i) => {
                const hasL = i < before.length, hasR = i < after.length
                const lNum = hasL && block.before_lines ? block.before_lines[0] + i : null
                const rNum = hasR && block.after_lines ? block.after_lines[0] + i : null
                return <SplitLine key={i}
                  leftNum={hasL ? lNum : null} rightNum={hasR ? rNum : null}
                  leftType={hasL ? 'removed' : 'empty'} rightType={hasR ? 'added' : 'empty'}
                  leftHtml={hasL ? tokens.base[lNum - 1] : null} rightHtml={hasR ? tokens.head[rNum - 1] : null}
                  leftFallback={hasL ? before[i] : null} rightFallback={hasR ? after[i] : null} />
              })}
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

export function FileCard({ file, viewMode }) {
  const [collapsed, setCollapsed] = useState(() => {
    let total = 0
    for (const b of file.blocks_to_show) {
      if (b.type === 'hunk') total += (b.before_content?.split('\n').length ?? 0) + (b.after_content?.split('\n').length ?? 0)
    }
    return total >= 200
  })

  const tokens = useFileHighlight({
    path: file.path, lang: file.lang,
    baseContent: file.base_content, headContent: file.head_content,
  })

  const parts = file.path.split('/')
  const filename = parts.pop()
  const dir = parts.join('/')

  return (
    <div style={{
      border: '1px solid var(--color-border)',
      borderRadius: 6,
      marginBottom: 12,
      overflow: 'hidden',
    }}>
      {/* File header */}
      <div
        onClick={() => setCollapsed(c => !c)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '7px 12px',
          background: 'var(--bg-elevated)',
          borderBottom: collapsed ? 'none' : '1px solid var(--color-border)',
          cursor: 'pointer', userSelect: 'none',
          fontFamily: 'var(--font-mono)', fontSize: 13,
        }}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-component)'}
        onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-elevated)'}
      >
        <span style={{
          fontSize: 9, color: 'var(--text-disabled)',
          display: 'inline-block',
          transform: collapsed ? 'rotate(0deg)' : 'rotate(90deg)',
          transition: 'transform 100ms',
        }}>▶</span>
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{filename}</span>
        {dir && <span style={{ color: 'var(--text-secondary)' }}>{dir}</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, fontSize: 12 }}>
          <span style={{ color: 'var(--text-green)' }}>+{file.additions}</span>
          <span style={{ color: 'var(--text-red)' }}>-{file.deletions}</span>
        </div>
      </div>

      {!collapsed && (
        <div style={{ overflowX: 'auto', background: 'var(--bg-diff)' }}>
          {viewMode === 'unified'
            ? <UnifiedDiff file={file} tokens={tokens} />
            : <SplitDiff file={file} tokens={tokens} />
          }
        </div>
      )}
    </div>
  )
}
