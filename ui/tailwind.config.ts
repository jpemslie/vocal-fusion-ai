import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:       "#0a0a0a",
        surface:  "#111111",
        surface2: "#161616",
        border:   "#222222",
        accent:   "#7c3aed",
        accent2:  "#a855f7",
        muted:    "#555555",
        muted2:   "#888888",
        good:     "#22c55e",
        warn:     "#f59e0b",
        err:      "#ef4444",
      },
      fontFamily: {
        mono: ["SF Mono", "Fira Code", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
