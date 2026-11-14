import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// El build va a app/static: FastAPI sirve index.html en / y los assets en /static.
const API = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: { outDir: '../static', emptyOutDir: true },
  server: {
    proxy: Object.fromEntries(
      ['/analyze', '/models', '/health', '/metrics'].map((p) => [p, API]),
    ),
  },
})
