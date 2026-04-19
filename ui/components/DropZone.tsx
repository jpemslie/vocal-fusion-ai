"use client";
import { useRef, useState, DragEvent } from "react";

interface Props {
  label: string;
  slot: "A" | "B";
  file: File | null;
  onChange: (f: File) => void;
}

/* Tailwind needs static class strings — no template literals with dynamic color names */
const SLOT_STYLES = {
  A: {
    base:      "border-white/10 hover:border-violet-500/40 hover:bg-violet-500/5",
    over:      "border-violet-400 bg-violet-500/10 drop-active",
    loaded:    "border-violet-500/60 bg-violet-500/5",
    ring:      "ring-violet-500/30",
    badge:     "bg-violet-500 text-white",
    badgeDim:  "bg-white/5 text-white/30",
    check:     "bg-violet-500/20",
    checkPath: "#8b5cf6",
    label:     "text-violet-400",
    icon:      "#8b5cf6",
    iconBg:    "bg-violet-500/10",
  },
  B: {
    base:      "border-white/10 hover:border-fuchsia-500/40 hover:bg-fuchsia-500/5",
    over:      "border-fuchsia-400 bg-fuchsia-500/10 drop-active",
    loaded:    "border-fuchsia-500/60 bg-fuchsia-500/5",
    ring:      "ring-fuchsia-500/30",
    badge:     "bg-fuchsia-500 text-white",
    badgeDim:  "bg-white/5 text-white/30",
    check:     "bg-fuchsia-500/20",
    checkPath: "#d946ef",
    label:     "text-fuchsia-400",
    icon:      "#d946ef",
    iconBg:    "bg-fuchsia-500/10",
  },
};

export default function DropZone({ label, slot, file, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const s = SLOT_STYLES[slot];

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onChange(f);
  }

  const hasFile = !!file;
  const borderClass = over ? s.over : hasFile ? s.loaded : s.base;

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
      className={[
        "relative rounded-2xl border-2 p-6 cursor-pointer transition-all duration-200 select-none overflow-hidden",
        "flex flex-col gap-3 min-h-[148px] justify-center",
        borderClass,
        over ? "scale-[1.02]" : "",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.aac,.m4a"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onChange(f); }}
      />

      {/* slot badge */}
      <div className={[
        "absolute top-3 right-3 w-6 h-6 rounded-full flex items-center justify-center text-[0.6rem] font-bold",
        hasFile ? s.badge : s.badgeDim,
      ].join(" ")}>
        {slot}
      </div>

      {hasFile ? (
        <>
          <div className={`w-10 h-10 rounded-full ${s.check} flex items-center justify-center shrink-0`}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 10l5 5 7-9" stroke={s.checkPath} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="flex flex-col gap-0.5 min-w-0">
            <span className={`text-[0.6rem] uppercase tracking-widest ${s.label}`}>{label}</span>
            <span className="text-[0.75rem] text-white font-medium truncate leading-snug">
              {file.name.replace(/\.[^/.]+$/, "")}
            </span>
            <span className="text-[0.58rem] text-white/30">
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </span>
          </div>
        </>
      ) : (
        <>
          <div className={[
            `w-10 h-10 rounded-full ${s.iconBg} flex items-center justify-center transition-transform shrink-0`,
            over ? "scale-110" : "",
          ].join(" ")}>
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
              <path d="M11 3v12M7 9l4-4 4 4" stroke={s.icon} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M4 17h14" stroke={s.icon} strokeWidth="2" strokeLinecap="round" opacity=".4"/>
            </svg>
          </div>
          <div className="flex flex-col gap-1">
            <span className={`text-[0.6rem] uppercase tracking-widest ${s.label} opacity-70`}>{label}</span>
            <span className="text-[0.7rem] text-white/30">
              {over ? "Drop it!" : "Drop MP3 or click"}
            </span>
          </div>
        </>
      )}

      {hasFile && (
        <div className={`absolute inset-0 rounded-2xl pointer-events-none ring-1 ${s.ring}`} />
      )}
    </div>
  );
}
