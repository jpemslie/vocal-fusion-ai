"use client";
import { useEffect, useRef, useState } from "react";

interface Props {
  variants: { radio?: string; club?: string; intimate?: string };
  outputUrl?: string;
}

type Tab = "main" | "radio" | "club" | "intimate";

const TAB_META: Record<Tab, { label: string; emoji: string; desc: string }> = {
  main:     { label: "Main Mix",  emoji: "🎵", desc: "Full fusion" },
  radio:    { label: "Radio",     emoji: "📻", desc: "Clean & punchy" },
  club:     { label: "Club",      emoji: "🔊", desc: "Bass boosted" },
  intimate: { label: "Intimate",  emoji: "🎧", desc: "Stripped back" },
};

export default function WavePlayer({ variants, outputUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef        = useRef<import("wavesurfer.js").default | null>(null);
  const [active, setActive]       = useState<Tab>("main");
  const [playing, setPlaying]     = useState(false);
  const [ready, setReady]         = useState(false);
  const [current, setCurrent]     = useState(0);
  const [duration, setDuration]   = useState(0);

  const tabs: Tab[] = [
    ...(outputUrl ? ["main" as Tab] : []),
    ...(["radio","club","intimate"] as Tab[]).filter(t => !!(variants as Record<string, string | undefined>)[t]),
  ];

  const activeUrl = active === "main"
    ? outputUrl
    : (variants as Record<string, string | undefined>)[active];

  useEffect(() => {
    if (!containerRef.current || !activeUrl) return;
    let dead = false;

    import("wavesurfer.js").then(({ default: WaveSurfer }) => {
      if (dead || !containerRef.current) return;
      wsRef.current?.destroy();
      wsRef.current = null;
      setReady(false); setPlaying(false); setCurrent(0); setDuration(0);

      const ws = WaveSurfer.create({
        container:     containerRef.current,
        waveColor:     "#3b0764",
        progressColor: "#a855f7",
        cursorColor:   "#c084fc",
        barWidth:      3,
        barGap:        2,
        barRadius:     3,
        height:        80,
        normalize:     true,
        interact:      true,
        url:           activeUrl,
      });

      ws.on("ready",        () => { setDuration(ws.getDuration()); setReady(true); });
      ws.on("audioprocess", () => setCurrent(ws.getCurrentTime()));
      ws.on("seeking",      () => setCurrent(ws.getCurrentTime()));
      ws.on("play",         () => setPlaying(true));
      ws.on("pause",        () => setPlaying(false));
      ws.on("finish",       () => { setPlaying(false); setCurrent(0); });

      wsRef.current = ws;
    });

    return () => { dead = true; wsRef.current?.destroy(); wsRef.current = null; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeUrl]);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2,"0")}`;
  const pct  = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <div className="flex flex-col gap-4 animate-slide-up">
      {/* tabs */}
      {tabs.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {tabs.map(t => {
            const m = TAB_META[t];
            return (
              <button
                key={t}
                onClick={() => setActive(t)}
                className={[
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[0.62rem] uppercase tracking-wider transition-all duration-200",
                  active === t
                    ? "bg-violet-600 text-white shadow-lg shadow-violet-500/20"
                    : "bg-white/5 text-muted hover:bg-white/10 hover:text-white border border-white/10",
                ].join(" ")}
              >
                <span>{m.emoji}</span>
                <span>{m.label}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* waveform card */}
      <div className="rounded-2xl overflow-hidden bg-[#0d0d1a] border border-violet-500/20 relative">
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="flex items-end gap-1">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className={`w-1.5 rounded-full bg-violet-500/60 bar${i+1}`}
                  style={{ height: 8 }}
                />
              ))}
            </div>
          </div>
        )}
        <div ref={containerRef} className="px-4 pt-4 pb-2" />

        {/* time scrub bar */}
        {ready && (
          <div className="px-4 pb-3">
            <div className="h-0.5 bg-white/5 rounded-full overflow-hidden">
              <div className="h-full bg-violet-500/60 transition-all duration-100" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* transport */}
      <div className="flex items-center gap-4">
        {/* play/pause */}
        <button
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          className={[
            "w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 shrink-0",
            ready
              ? "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-500/30 hover:scale-105 active:scale-95"
              : "bg-white/5 text-muted cursor-not-allowed",
          ].join(" ")}
        >
          {playing ? (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <rect x="2" y="2" width="5" height="12" rx="1.5"/>
              <rect x="9" y="2" width="5" height="12" rx="1.5"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M4 2.5l10 5.5-10 5.5V2.5z"/>
            </svg>
          )}
        </button>

        {/* time */}
        <div className="flex flex-col gap-0.5">
          <span className="text-[0.75rem] text-white tabular-nums font-mono">
            {fmt(current)}
          </span>
          <span className="text-[0.58rem] text-muted2 tabular-nums">
            / {fmt(duration)}
          </span>
        </div>

        {/* active tab info */}
        <div className="flex flex-col gap-0.5 flex-1">
          <span className="text-[0.6rem] uppercase tracking-widest text-violet-400">
            {TAB_META[active].label}
          </span>
          <span className="text-[0.58rem] text-muted2">
            {TAB_META[active].desc}
          </span>
        </div>

        {/* download */}
        {activeUrl && (
          <a
            href={activeUrl}
            download
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[0.62rem] text-muted hover:text-white transition-all uppercase tracking-wider"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <path d="M6 1v7M3 6l3 3 3-3M1 11h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
            </svg>
            Save
          </a>
        )}
      </div>
    </div>
  );
}
