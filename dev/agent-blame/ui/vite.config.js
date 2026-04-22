import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Divergent copy of pr-bot/ui/vite.config.js.
// Uses 127.0.0.1 explicitly — `localhost` resolves to ::1 on macOS and our
// Python stdlib HTTPServer binds v4-only, so the proxy silently returns
// empty bodies if we leave `localhost` here.
// API_PORT env var can override the backend port; the launcher threads
// this through so the two processes always agree.

const apiPort = process.env.API_PORT || '7979'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': `http://127.0.0.1:${apiPort}`,
    },
  },
})
