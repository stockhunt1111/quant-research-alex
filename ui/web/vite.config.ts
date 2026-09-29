import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes dist/, which the server serves at /; `npm run dev` serves the sources with the API of a server
// running on its default port (python -m server).
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, sourcemap: true },
  server: { proxy: { "/api": { target: "http://127.0.0.1:8600", changeOrigin: false } } },
});
