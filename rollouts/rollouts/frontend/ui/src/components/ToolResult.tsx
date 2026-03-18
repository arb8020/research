interface ToolResultProps {
  result: string
}

export function ToolResult({ result }: ToolResultProps) {
  const isLong = result.length > 500
  const displayText = isLong ? result.slice(0, 500) + '\n… (truncated)' : result

  return (
    <div className="my-1 min-w-0 overflow-hidden">
      <div
        className="text-xs font-mono whitespace-pre-wrap overflow-x-auto"
        style={{
          color: 'var(--color-dark-text-secondary)',
          maxHeight: '300px',
          overflowY: 'auto',
        }}
      >
        {displayText}
      </div>
    </div>
  )
}
