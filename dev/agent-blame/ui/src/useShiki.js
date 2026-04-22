// Divergent copy of pr-bot/ui/src/useShiki.js.
// Same highlighter singleton; agent-blame adds useSingleFileHighlight for
// whole-file viewing (pr-bot highlights base + head separately).

import { useEffect, useRef, useState } from 'react'
import { createHighlighter } from 'shiki'

// Singleton highlighter promise — created once, reused
let highlighterPromise = null
function getHighlighter() {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ['github-dark'],
      langs: ['python', 'javascript', 'typescript', 'tsx', 'jsx', 'go', 'rust',
              'java', 'c', 'cpp', 'bash', 'shell', 'ruby', 'php', 'css', 'html',
              'json', 'yaml', 'toml', 'markdown', 'sql', 'text'],
    })
  }
  return highlighterPromise
}

// Lightweight language inference by extension. Shiki falls back to 'text'
// if we guess wrong, so this is best-effort.
export function inferLang(path) {
  const ext = path.split('.').pop()?.toLowerCase()
  const map = {
    py: 'python', pyi: 'python',
    js: 'javascript', mjs: 'javascript', cjs: 'javascript',
    ts: 'typescript', tsx: 'tsx', jsx: 'jsx',
    go: 'go', rs: 'rust',
    java: 'java', c: 'c', h: 'c', cc: 'cpp', cpp: 'cpp', hpp: 'cpp',
    sh: 'bash', bash: 'bash', zsh: 'bash',
    rb: 'ruby', php: 'php',
    css: 'css', html: 'html', htm: 'html',
    json: 'json', yaml: 'yaml', yml: 'yaml', toml: 'toml',
    md: 'markdown', markdown: 'markdown',
    sql: 'sql',
  }
  return map[ext] || 'text'
}

// Highlight one file's content; return per-line HTML strings.
export function useSingleFileHighlight({ path, content }) {
  const [lines, setLines] = useState([])
  const cacheRef = useRef(new Map())

  useEffect(() => {
    if (!content) { setLines([]); return }
    const lang = inferLang(path)
    const key = `${path}::${lang}::${content.length}::${content.slice(0, 32)}`
    if (cacheRef.current.has(key)) { setLines(cacheRef.current.get(key)); return }

    let cancelled = false
    getHighlighter().then(hl => {
      if (cancelled) return
      const html = hl.codeToHtml(content, { lang, theme: 'github-dark' })
      const div = document.createElement('div')
      div.innerHTML = html
      const result = Array.from(div.querySelectorAll('code .line')).map(el => el.innerHTML)
      if (!cancelled) { cacheRef.current.set(key, result); setLines(result) }
    })
    return () => { cancelled = true }
  }, [path, content])

  return lines
}

// Highlight a full file's content and return per-line HTML arrays
// Returns { base: string[], head: string[] }
export function useFileHighlight({ path, lang, baseContent, headContent }) {
  const [tokens, setTokens] = useState({ base: [], head: [] })
  const cacheRef = useRef(new Map())

  useEffect(() => {
    if (!baseContent && !headContent) return

    const cacheKey = `${path}::${lang}::${baseContent?.length}::${headContent?.length}`
    if (cacheRef.current.has(cacheKey)) {
      setTokens(cacheRef.current.get(cacheKey))
      return
    }

    let cancelled = false
    getHighlighter().then(hl => {
      if (cancelled) return

      function extractLines(content) {
        if (!content) return []
        const html = hl.codeToHtml(content, { lang: lang || 'text', theme: 'github-dark' })
        const div = document.createElement('div')
        div.innerHTML = html
        return Array.from(div.querySelectorAll('code .line')).map(el => el.innerHTML)
      }

      const result = {
        base: extractLines(baseContent),
        head: extractLines(headContent),
      }
      if (!cancelled) {
        cacheRef.current.set(cacheKey, result)
        setTokens(result)
      }
    })

    return () => { cancelled = true }
  }, [path, lang, baseContent, headContent])

  return tokens
}
