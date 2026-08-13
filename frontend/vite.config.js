import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' -> assets referenced relatively, so the built app works served from any mount
// point (behind Caddy at / or ngrok). Build to dist/, which run_all.sh serves on :8000.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
})
