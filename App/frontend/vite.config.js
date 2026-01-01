import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Pour garder le même port qu'avant si tu veux
    open: true  // Ouvre le navigateur automatiquement au lancement
  }
})