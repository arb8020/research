import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  label?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error(`[ErrorBoundary:${this.props.label ?? 'unknown'}]`, error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          className="p-4 rounded font-mono text-xs"
          style={{
            background: 'var(--color-dark-elevated)',
            border: '1px solid #ef4444',
            color: '#ef4444',
          }}
        >
          <div className="font-semibold mb-2">
            {this.props.label ? `[${this.props.label}] ` : ''}Runtime error
          </div>
          <pre className="whitespace-pre-wrap" style={{ color: 'var(--color-dark-text-secondary)' }}>
            {this.state.error.message}
            {'\n\n'}
            {this.state.error.stack?.split('\n').slice(0, 8).join('\n')}
          </pre>
          <button
            className="mt-3 px-2 py-1 rounded text-xs hover:opacity-80"
            style={{ border: '1px solid #ef4444', color: '#ef4444' }}
            onClick={() => this.setState({ error: null })}
          >
            Dismiss
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
