"use client";
import type { SpectralMetadata, AttemptSummary } from "../lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  chartScore:  number;
  chartGrade:  string;
  mixScore:    number;
  mixGrade:    string;
  metadata?:   SpectralMetadata;
  attempts?:   AttemptSummary[];
  seed?:       number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function gradeColor(grade: string): string {
  const g = grade[0];
  if (g === "S") return "text-yellow-400";
  if (g === "A") return "text-violet-400";
  if (g === "B") return "text-blue-400";
  if (g === "C") return "text-teal-400";
  return "text-muted2";
}

// Band delta bar — positive = above ref (generally good for presence/air),
// negative = below (lacking). Range: -20 to +20 dB.
function DeltaBar({ label, value }: { label: string; value: number }) {
  const clamped  = Math.max(-20, Math.min(20, value));
  const pct      = Math.abs(clamped) / 20 * 100;
  const isPos    = clamped >= 0;
  const barColor = Math.abs(clamped) <= 2
    ? "bg-emerald-500"
    : Math.abs(clamped) <= 8
      ? "bg-amber-500"
      : "bg-red-500";

  return (
    <div className="flex items-center gap-2 text-[0.6rem]">
      <span className="w-[52px] text-muted2 shrink-0 tabular-nums text-right">{label}</span>
      {/* left side (negative) */}
      <div className="flex-1 flex justify-end h-1.5 relative">
        <div className="absolute inset-0 bg-white/5 rounded-full" />
        {!isPos && (
          <div
            className={`absolute right-0 top-0 h-full rounded-full ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      {/* center line */}
      <div className="w-px h-3 bg-white/20 shrink-0" />
      {/* right side (positive) */}
      <div className="flex-1 h-1.5 relative">
        <div className="absolute inset-0 bg-white/5 rounded-full" />
        {isPos && (
          <div
            className={`absolute left-0 top-0 h-full rounded-full ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span className={`w-[36px] tabular-nums ${clamped >= 0 ? "text-emerald-400" : "text-red-400"} shrink-0`}>
        {clamped >= 0 ? "+" : ""}{clamped.toFixed(1)}
      </span>
    </div>
  );
}

// Sibilance meter — target range: 0.15–0.40
function SibilanceMeter({ value }: { value: number }) {
  const MAX = 1.0;
  const pct = Math.min(value / MAX * 100, 100);
  const lo  = 0.15 / MAX * 100;
  const hi  = 0.40 / MAX * 100;

  const inRange = value >= 0.15 && value <= 0.40;
  const barColor = inRange ? "bg-emerald-500" : value < 0.15 ? "bg-blue-500" : "bg-red-500";
  const label    = inRange ? "Perfect" : value < 0.15 ? "Under-essed" : "Too harsh";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-[0.6rem]">
        <span className="text-muted2">Sibilance</span>
        <span className={inRange ? "text-emerald-400" : "text-amber-400"}>
          {value.toFixed(2)} · {label}
        </span>
      </div>
      {/* track */}
      <div className="relative h-2 rounded-full bg-white/5 overflow-hidden">
        {/* target zone */}
        <div
          className="absolute top-0 h-full bg-emerald-500/20 rounded-sm"
          style={{ left: `${lo}%`, width: `${hi - lo}%` }}
        />
        {/* current value */}
        <div
          className={`absolute left-0 top-0 h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-[0.52rem] text-muted2/60">
        <span>0</span>
        <span>↑ target 0.15–0.40</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type DeltaKey = "delta_sub" | "delta_bass" | "delta_lo_mid" | "delta_mid" | "delta_hi_mid" | "delta_presence" | "delta_air";
const BANDS: Array<{ key: DeltaKey; label: string }> = [
  { key: "delta_sub",      label: "Sub"      },
  { key: "delta_bass",     label: "Bass"     },
  { key: "delta_lo_mid",   label: "Lo-Mid"   },
  { key: "delta_mid",      label: "Mid"      },
  { key: "delta_hi_mid",   label: "Hi-Mid"   },
  { key: "delta_presence", label: "Presence" },
  { key: "delta_air",      label: "Air"      },
];

export default function ScoreCard({
  chartScore, chartGrade, mixScore, mixGrade, metadata, attempts, seed,
}: Props) {
  const cColor = gradeColor(chartGrade);
  const mColor = gradeColor(mixGrade);

  return (
    <div className="flex flex-col gap-4 animate-slide-up">

      {/* ── Score row ── */}
      <div className="grid grid-cols-2 gap-3">
        {/* chart score */}
        <div className="flex flex-col items-center gap-1 py-4 rounded-2xl bg-white/[0.03] border border-white/5">
          <span className="text-[0.55rem] uppercase tracking-widest text-muted2">Chart Score</span>
          <span className={`text-4xl font-mono font-bold tabular-nums ${cColor}`}>
            {chartScore}
            <span className="text-base text-muted2">/100</span>
          </span>
          <span className={`text-[0.62rem] uppercase tracking-widest font-semibold ${cColor}`}>
            {chartGrade}
          </span>
        </div>

        {/* mix score */}
        <div className="flex flex-col items-center gap-1 py-4 rounded-2xl bg-white/[0.03] border border-white/5">
          <span className="text-[0.55rem] uppercase tracking-widest text-muted2">Mix Quality</span>
          <span className={`text-4xl font-mono font-bold tabular-nums ${mColor}`}>
            {mixScore}
            <span className="text-base text-muted2">/100</span>
          </span>
          <span className={`text-[0.62rem] uppercase tracking-widest font-semibold ${mColor}`}>
            {mixGrade}
          </span>
        </div>
      </div>

      {/* ── Spectral panel ── */}
      {metadata && (
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 p-4 flex flex-col gap-4">
          {/* band delta bars */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-[0.6rem] uppercase tracking-widest text-muted2">
                Spectral Balance
              </span>
              <span className="text-[0.52rem] text-muted2/50">vs reference · dB</span>
            </div>
            {BANDS.map(({ key, label }) => (
              <DeltaBar
                key={key}
                label={label}
                value={metadata[key] ?? 0}
              />
            ))}
          </div>

          {/* divider */}
          <div className="h-px bg-white/5" />

          {/* sibilance meter */}
          <SibilanceMeter value={metadata.sibilance_ratio ?? 0.25} />

          {/* LUFS + DR row */}
          <div className="grid grid-cols-2 gap-2 text-[0.6rem]">
            <div className="flex flex-col gap-0.5 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-muted2">Loudness (LUFS)</span>
              <span className="text-white tabular-nums font-mono">
                {metadata.lufs?.toFixed(1) ?? "—"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-muted2">Dynamic Range</span>
              <span className="text-white tabular-nums font-mono">
                {metadata.dynamic_range?.toFixed(1) ?? "—"} dB
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Refinement history ── */}
      {attempts && attempts.length > 1 && (
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 p-4 flex flex-col gap-3">
          <span className="text-[0.6rem] uppercase tracking-widest text-muted2">
            Refinement Attempts
          </span>
          <div className="flex flex-col gap-1.5">
            {attempts.map((att) => {
              const isMax = att.chart_score === Math.max(...attempts.map(a => a.chart_score));
              return (
                <div
                  key={att.n}
                  className={[
                    "flex items-center gap-2 px-3 py-2 rounded-xl text-[0.62rem] border",
                    isMax
                      ? "bg-violet-500/10 border-violet-500/20"
                      : "bg-white/[0.02] border-white/5",
                  ].join(" ")}
                >
                  <span className="text-muted2 shrink-0">#{att.n}</span>
                  <span className={`font-mono font-semibold tabular-nums ${gradeColor(att.chart_grade)}`}>
                    {att.chart_score}/100
                  </span>
                  <span className={gradeColor(att.chart_grade)}>{att.chart_grade}</span>
                  {Object.keys(att.overrides).length > 0 && (
                    <span className="text-muted2/60 truncate ml-auto">
                      {Object.entries(att.overrides)
                        .map(([k, v]) => `${k.replace(/_/g, " ")} ${v > 0 ? "+" : ""}${v}`)
                        .join(" · ")}
                    </span>
                  )}
                  {isMax && (
                    <span className="ml-auto shrink-0 text-violet-400">best</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* seed */}
      {seed != null && (
        <div className="text-center text-[0.52rem] text-muted2/40 tabular-nums">
          seed {seed}
        </div>
      )}
    </div>
  );
}
