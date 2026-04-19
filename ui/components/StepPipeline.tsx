"use client";
import { Step } from "../lib/api";

interface Props {
  steps: Step[];
  message?: string;
}

const STEP_ICONS: Record<number, string> = {
  1:  "🎵",
  2:  "🔬",
  3:  "🥁",
  4:  "⚡",
  5:  "🎚",
  6:  "🎛",
  7:  "✨",
  8:  "📊",
  9:  "🎁",
  10: "🚀",
};

export default function StepPipeline({ steps, message }: Props) {
  if (!steps.length) return null;

  const done   = steps.filter(s => s.status === "done").length;
  const total  = steps.length;

  return (
    <div className="flex flex-col gap-3">
      {/* mini progress summary */}
      <div className="flex items-center justify-between text-[0.6rem] text-muted2 uppercase tracking-widest">
        <span>{message ?? "Processing…"}</span>
        <span>{done}/{total} steps</span>
      </div>

      {/* steps */}
      <div className="flex flex-col gap-px">
        {steps.map((s) => {
          const isRunning = s.status === "running";
          const isDone    = s.status === "done";
          const isError   = s.status === "error";

          return (
            <div
              key={s.n}
              className={[
                "flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-300",
                isRunning ? "bg-violet-500/10 border border-violet-500/20" : "border border-transparent",
              ].join(" ")}
            >
              {/* icon / spinner */}
              <div className="w-5 shrink-0 flex items-center justify-center">
                {isRunning ? (
                  <div className="w-3.5 h-3.5 rounded-full border-2 border-violet-400 border-t-transparent animate-spin-fast" />
                ) : isDone ? (
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <circle cx="7" cy="7" r="6" fill="#22c55e" opacity=".2"/>
                    <path d="M4 7l2.5 2.5L10 5" stroke="#22c55e" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : isError ? (
                  <span className="text-red-400 text-xs">✕</span>
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-white/10" />
                )}
              </div>

              {/* emoji */}
              <span className={["text-sm transition-opacity", isDone || isRunning ? "opacity-100" : "opacity-20"].join(" ")}>
                {STEP_ICONS[s.n] ?? "◦"}
              </span>

              {/* title */}
              <span className={[
                "flex-1 text-[0.68rem] transition-colors duration-300",
                isRunning ? "text-white" : isDone ? "text-white/50" : isError ? "text-red-400" : "text-white/20",
              ].join(" ")}>
                {s.title}
              </span>

              {/* elapsed */}
              {s.elapsed != null && isDone && (
                <span className="text-[0.58rem] text-muted2 tabular-nums shrink-0">
                  {s.elapsed < 60 ? `${s.elapsed.toFixed(1)}s` : `${Math.floor(s.elapsed / 60)}m${(s.elapsed % 60 | 0)}s`}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
