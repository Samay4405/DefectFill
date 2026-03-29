import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["echarts", "echarts-for-react"],
  },
};

export default nextConfig;
