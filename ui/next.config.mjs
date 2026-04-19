const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/fuse",                   destination: `${BACKEND}/fuse` },
      { source: "/status/:id",             destination: `${BACKEND}/status/:id` },
      { source: "/download/:id",           destination: `${BACKEND}/download/:id` },
      { source: "/download/:id/:variant",  destination: `${BACKEND}/download/:id/:variant` },
      { source: "/health",                 destination: `${BACKEND}/health` },
    ];
  },
};

export default nextConfig;
