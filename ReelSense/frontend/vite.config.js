import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
     
      '/recommendations': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
     
      '/root': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});