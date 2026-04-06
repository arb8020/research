import { useEffect, useRef } from 'react'
import { EditorView, keymap, lineNumbers, drawSelection, highlightActiveLine, scrollPastEnd } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { json } from '@codemirror/lang-json'
import { yaml } from '@codemirror/lang-yaml'
import { foldKeymap, indentOnInput, syntaxHighlighting, defaultHighlightStyle, bracketMatching, foldGutter } from '@codemirror/language'
import { defaultKeymap } from '@codemirror/commands'
import { vscodeDark } from '@uiw/codemirror-theme-vscode'

const baseTheme = EditorView.theme({
  '&': {
    fontSize: '11px',
    fontFamily: '"IBM Plex Mono", monospace',
    background: '#0a0a0a',
    borderRadius: '2px',
    border: '1px solid #1e1e1e',
    height: '100%',
    minHeight: '200px',
  },
  '.cm-scroller': {
    overflow: 'auto',
    lineHeight: '1.5',
  },
  '.cm-content': {
    padding: '0.75rem 0',
    caretColor: 'transparent',
  },
  '.cm-foldGutter': {
    width: '14px',
  },
  '.cm-foldGutter .cm-gutterElement': {
    padding: '0 2px',
    cursor: 'pointer',
    color: '#444',
    userSelect: 'none',
  },
  '.cm-foldGutter .cm-gutterElement:hover': {
    color: '#888',
  },
  '.cm-gutters': {
    background: '#0a0a0a',
    border: 'none',
    borderRight: '1px solid #1a1a1a',
  },
  '.cm-lineNumbers .cm-gutterElement': {
    color: '#333',
    minWidth: '2.5em',
    padding: '0 6px 0 4px',
  },
  '.cm-activeLine': {
    background: 'transparent',
  },
  '.cm-activeLineGutter': {
    background: 'transparent',
  },
})

export function CodeMirrorViewer({
  value,
  lang,
  onViewReady,
}: {
  value: string
  lang: 'json' | 'yaml'
  onViewReady?: (view: EditorView) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const langExt = lang === 'json' ? json() : yaml()

    const state = EditorState.create({
      doc: value,
      extensions: [
        vscodeDark,
        baseTheme,
        langExt,
        lineNumbers(),
        foldGutter({
          markerDOM(open: boolean) {
            const el = document.createElement('span')
            el.textContent = open ? '▾' : '▸'
            el.style.fontSize = '10px'
            return el
          },
        }),
        drawSelection(),
        bracketMatching(),
        indentOnInput(),
        syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
        highlightActiveLine(),
        scrollPastEnd(),
        keymap.of([...defaultKeymap, ...foldKeymap]),
        EditorState.readOnly.of(true),
        EditorView.lineWrapping,
      ],
    })

    const view = new EditorView({ state, parent: containerRef.current })
    onViewReady?.(view)

    return () => { view.destroy() }
  }, [value, lang]) // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={containerRef} style={{ height: '100%', minHeight: 200 }} />
}

