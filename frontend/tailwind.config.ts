import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        surface: "#f4f6f5",
        panel: "#e8ece8",
        ink: "#1f2a28",
        accent: "#2f7066",
        warning: "#f45b3d"
      },
      boxShadow: {
        soft: "0 12px 30px rgba(31, 42, 40, 0.08)"
      }
    },
  },
  plugins: [],
};

export default config;
