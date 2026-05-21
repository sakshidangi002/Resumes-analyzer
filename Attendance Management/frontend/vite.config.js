import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/resume-api": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/resume": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/portal.html": { target: "http://127.0.0.1:5001", changeOrigin: true },
        },
    },
    preview: {
        port: 4173,
        proxy: {
            "/api": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/resume-api": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/resume": { target: "http://127.0.0.1:5001", changeOrigin: true },
            "/portal.html": { target: "http://127.0.0.1:5001", changeOrigin: true },
        },
    },
    build: {
        outDir: "../backend/frontend_build",
        emptyOutDir: true,
        sourcemap: false,
    },
});
