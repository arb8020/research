import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './globals.css'
import App from './App.tsx'
import { Gsm8kCalculatorPanel } from './plugins/Gsm8kCalculatorPanel'
import { registerByEnvKind, registerByEvalName } from './plugins/registry'

registerByEvalName('gsm8k_progress_demo', {
  label: 'Calculator',
  render: (props) => <Gsm8kCalculatorPanel {...props} />,
})
registerByEnvKind('calculator', {
  label: 'Calculator',
  render: (props) => <Gsm8kCalculatorPanel {...props} />,
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
