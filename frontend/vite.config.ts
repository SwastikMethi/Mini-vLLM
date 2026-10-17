import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const contracts = fileURLToPath(new URL('../mini_vllm_docs/contracts', import.meta.url))

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@contracts': contracts } },
  server: {
    fs: { allow: ['.', contracts] },
    proxy: { '/api': `http://127.0.0.1:${process.env.MINI_VLLM_PORT ?? 8077}` },
  },
})
