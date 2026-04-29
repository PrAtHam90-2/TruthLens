import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      // Proxy all /api requests to the FastAPI backend
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
