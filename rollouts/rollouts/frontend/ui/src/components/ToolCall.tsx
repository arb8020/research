import { CodeBlock } from './CodeBlock'

interface ToolCallProps {
  name: string
  args: unknown
  id?: string
}

function isCodeLike(text: string): boolean {
  if (!text || typeof text !== 'string' || text.length < 10) return false

  const codeIndicators = [
    'import ', 'from ', 'def ', 'class ', 'return ', 'print(',
    'torch.', 'Path(', 'subprocess', 'sys.', 'os.',
    '#include', 'function ', 'const ', 'let ', 'var ', 'export ', 'import {',
    'python -', 'bash -c', 'grep -', 'if (', 'for (', 'while (',
    'try {', 'catch {', 'async ', 'await ',
  ]

  const hasCodeIndicators = codeIndicators.some(indicator => text.includes(indicator))
  const hasCodeStructure = (
    (text.includes('(') && text.includes(')')) ||
    (text.includes('[') && text.includes(']')) ||
    (text.includes('=') && text.includes('(')) ||
    text.includes('def ') || text.includes('function ') ||
    text.includes('import ') || text.includes('#include')
  )

  if (text.includes('\n') && (hasCodeIndicators || hasCodeStructure)) return true
  if (hasCodeIndicators && text.length > 50) return true
  return false
}

function detectLanguage(code: string): string {
  if (code.includes('#include') || code.includes('__global__') || code.includes('__device__')) return 'cpp'
  if (code.includes('import ') || code.includes('from ') || code.includes('def ') || code.includes('torch.')) return 'python'
  if (code.includes('function ') || code.includes('const ') || code.includes('export ')) return 'javascript'
  if (code.startsWith('#!/') || code.includes('echo ') || code.includes('grep ')) return 'bash'
  return 'text'
}

export function ToolCall({ name, args, id: _id }: ToolCallProps) {
  void _id
  if (typeof args === 'string') {
    return (
      <div className="my-1 min-w-0 overflow-hidden">
        <div className="font-medium mb-0.5 text-xs" style={{ color: 'var(--color-dark-text-secondary)' }}>
          {name}
        </div>
        {isCodeLike(args) ? (
          <CodeBlock code={args} language={detectLanguage(args)} />
        ) : (
          <pre className="whitespace-pre-wrap text-xs font-mono" style={{ color: 'var(--color-dark-text-secondary)' }}>{args}</pre>
        )}
      </div>
    )
  }

  if (!args || typeof args !== 'object') {
    return (
      <div className="my-1 min-w-0 overflow-hidden">
        <div className="font-medium mb-0.5 text-xs" style={{ color: 'var(--color-dark-text-secondary)' }}>
          {name}
        </div>
        <pre className="whitespace-pre-wrap text-xs font-mono">{JSON.stringify(args, null, 2)}</pre>
      </div>
    )
  }

  const argsObj = args as Record<string, unknown>

  // Find a code-like field to highlight
  const codeFields = ['code', 'content', 'command', 'file_content', 'path']
  let code: string | null = null
  let codeField: string | null = null

  for (const field of codeFields) {
    if (field in argsObj && typeof argsObj[field] === 'string') {
      const candidate = argsObj[field] as string
      if (isCodeLike(candidate)) {
        code = candidate
        codeField = field
        break
      }
    }
  }

  // Fallback: single long string field
  if (!code) {
    const stringFields = Object.entries(argsObj).filter(
      ([, v]) => typeof v === 'string' && (v as string).length > 50
    )
    if (stringFields.length === 1 && isCodeLike(stringFields[0][1] as string)) {
      code = stringFields[0][1] as string
      codeField = stringFields[0][0]
    }
  }

  const otherFields = Object.entries(argsObj).filter(([k]) => k !== codeField)
  const metaKeys = new Set(['timeout', 'id', 'tool_call_id'])
  const meta = otherFields.filter(([k]) => metaKeys.has(k.toLowerCase()))
  const body = otherFields.filter(([k]) => !metaKeys.has(k.toLowerCase()))

  return (
    <div className="my-1 min-w-0 overflow-hidden">
      <div className="flex items-center gap-2 mb-0.5">
        <span className="font-medium text-xs" style={{ color: 'var(--color-dark-text-secondary)' }}>
          {name}
        </span>
        {meta.length > 0 && (
          <span className="text-[10px]" style={{ color: 'var(--color-dark-text-muted)' }}>
            {meta.map(([k, v]) => `${k}: ${String(v)}`).join(' · ')}
          </span>
        )}
      </div>
      <div className="space-y-1 min-w-0 overflow-x-auto">
        {body.length > 0 && !code && (
          <div className="space-y-0.5">
            {body.map(([k, v]) => (
              <div key={k} className="text-xs">
                <span className="font-medium" style={{ color: 'var(--color-dark-text-secondary)' }}>{k}:</span>{' '}
                {typeof v === 'string' && isCodeLike(v) ? (
                  <CodeBlock code={v} language={detectLanguage(v)} />
                ) : (
                  <span style={{ color: 'var(--color-dark-text-muted)' }}>
                    {typeof v === 'string' ? v : JSON.stringify(v)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
        {code ? (
          <CodeBlock code={code} language={detectLanguage(code)} />
        ) : body.length === 0 && (
          <pre className="whitespace-pre-wrap text-xs font-mono overflow-x-auto">
            {JSON.stringify(argsObj, null, 2)}
          </pre>
        )}
      </div>
    </div>
  )
}
