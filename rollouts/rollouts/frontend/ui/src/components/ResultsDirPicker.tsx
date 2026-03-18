import { useState, useEffect, useRef } from 'react'
import { ChevronDown, FolderOpen } from 'lucide-react'
import { getResultsDirs, setResultsDir } from '../api'
import type { ResultsDirsResponse } from '../api'

interface ResultsDirPickerProps {
  onChanged: () => void
}

export function ResultsDirPicker({ onChanged }: ResultsDirPickerProps) {
  const [data, setData] = useState<ResultsDirsResponse | null>(null)
  const [open, setOpen] = useState(false)
  const [switching, setSwitching] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const load = () => {
    getResultsDirs()
      .then(setData)
      .catch(() => {})
  }

  useEffect(() => {
    load()
  }, [])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const handleSelect = async (path: string) => {
    if (path === data?.current) { setOpen(false); return }
    setSwitching(true)
    setOpen(false)
    try {
      await setResultsDir(path)
      load()
      onChanged()
    } catch {
      // ignore
    } finally {
      setSwitching(false)
    }
  }

  if (!data || data.dirs.length <= 1) return null

  const currentLabel = data.dirs.find(d => d.path === data.current)?.label ?? data.current.split('/').pop() ?? '?'

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        disabled={switching}
        className="flex items-center gap-1.5 px-2 py-1 rounded transition-opacity hover:opacity-80"
        style={{
          border: '1px solid var(--color-dark-border)',
          background: 'var(--color-dark-card)',
          color: 'var(--color-dark-text-secondary)',
          fontSize: 11,
          opacity: switching ? 0.5 : 1,
        }}
      >
        <FolderOpen className="h-3 w-3" />
        <span className="font-mono max-w-[160px] truncate">{currentLabel}</span>
        <ChevronDown className="h-3 w-3 opacity-60" />
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-1 rounded z-50 min-w-[220px]"
          style={{
            background: 'var(--color-dark-card)',
            border: '1px solid var(--color-dark-border)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          {data.dirs.map(d => (
            <button
              key={d.path}
              onClick={() => void handleSelect(d.path)}
              disabled={!d.exists}
              className="w-full text-left px-3 py-2 flex items-center gap-2 transition-colors"
              style={{
                color: d.path === data.current
                  ? 'var(--color-dark-text)'
                  : d.exists
                  ? 'var(--color-dark-text-secondary)'
                  : 'var(--color-dark-text-muted)',
                background: d.path === data.current ? 'var(--color-dark-elevated)' : undefined,
                fontSize: 12,
                opacity: d.exists ? 1 : 0.4,
              }}
              onMouseEnter={e => {
                if (d.path !== data.current && d.exists)
                  (e.currentTarget as HTMLElement).style.background = 'var(--color-dark-hover)'
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background =
                  d.path === data.current ? 'var(--color-dark-elevated)' : ''
              }}
            >
              {d.path === data.current && (
                <span style={{ color: 'var(--color-accent)', fontSize: 10 }}>●</span>
              )}
              <span className="font-mono truncate">{d.label}</span>
              {!d.exists && <span className="text-[10px] ml-auto opacity-50">missing</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
