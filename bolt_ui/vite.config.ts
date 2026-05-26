import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,      // 允许局域网和公网访问
    port: 5173,      // 可省略，默认就是5173
  },
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
});