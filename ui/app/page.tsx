"use client";
import { useState, useRef, useCallback } from "react";
import DropZone   from "../components/DropZone";
import WavePlayer from "../components/WavePlayer";
import ScoreCard  from "../components/ScoreCard";
import { startFuse, getStatus, Job } from "../lib/api";

type Phase = "idle" | "uploading" | "running" | "done" | "error";
const POLL_MS = 1500;

export default function Home() {
  const [songA,   setSongA]   = useState<File | null>(null);
  const [songB,   setSongB]   = useState<File | null>(null);
  const [phase,   setPhase]   = useState<Phase>("idle");
  const [job,     setJob]     = useState<Job | null>(null);
  const [err,     setErr]     = useState("");
  const [proMode, setProMode] = useState(true);
  const jobIdRef = useRef<string | null>(null);
  const pollRef  = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearPoll = () => {
    if (pollRef.current) { clearTimeout(pollRef.current); pollRef.current = null; }
  };

  const poll = useCallback(async () => {
    if (!jobIdRef.current) return;
    try {
      const j = await getStatus(jobIdRef.current);
      setJob(j);
      if (j.status === "done")  { setPhase("done");  return; }
      if (j.status === "error") { setPhase("error"); setErr(j.message ?? "Failed"); return; }
      pollRef.current = setTimeout(poll, POLL_MS);
    } catch {
      pollRef.current = setTimeout(poll, POLL_MS * 2);
    }
  }, []);

  async function handleFuse() {
    if (!songA || !songB) return;
    clearPoll();
    setPhase("uploading");
    setErr("");
    setJob(null);
    try {
      const id = await startFuse(songA, songB, false, 42, false, proMode);
      jobIdRef.current = id;
      setPhase("running");
      pollRef.current = setTimeout(poll, POLL_MS);
    } catch (e: unknown) {
      setPhase("error");
      setErr(e instanceof Error ? e.message : "Upload failed");
    }
  }

  function reset() {
    clearPoll();
    setPhase("idle");
    setJob(null);
    setErr("");
    jobIdRef.current = null;
    setSongA(null);
    setSongB(null);
  }

  const busy    = phase === "uploading" || phase === "running";
  const canFuse = !!songA && !!songB && !busy && phase !== "done";
  const attempts = job?.attempts ?? [];

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 gap-8 py-16">

      {/* background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[30%] w-[600px] h-[600px] rounded-full bg-violet-700/5 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[20%] w-[400px] h-[400px] rounded-full bg-fuchsia-700/5 blur-[100px]" />
      </div>

      {/* title */}
      <div className="flex items-center gap-3 z-10">
        <span className="text-2xl">🎤</span>
        <h1 className="text-2xl font-mono tracking-[0.3em] uppercase bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
          VocalFusion
        </h1>
        <span className="text-2xl">🥁</span>
      </div>

      {/* card */}
      <div className="w-full max-w-[480px] z-10 flex flex-col gap-4">

        {/* inputs */}
        <div className="grid grid-cols-2 gap-3">
          <DropZone label="Vocals" slot="A" file={songA} onChange={setSongA} />
          <DropZone label="Beat"   slot="B" file={songB} onChange={setSongB} />
        </div>

        {/* pro mode toggle — only visible when idle */}
        {phase === "idle" || phase === "error" ? (
          <button
            onClick={() => setProMode(p => !p)}
            className="flex items-center gap-2 self-start px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/10 text-[0.62rem] uppercase tracking-wider text-muted hover:text-white transition-all"
          >
            <div className={[
              "w-7 h-4 rounded-full relative transition-colors duration-200",
              proMode ? "bg-violet-600" : "bg-white/10",
            ].join(" ")}>
              <div className={[
                "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all duration-200",
                proMode ? "left-3.5" : "left-0.5",
              ].join(" ")} />
            </div>
            Professional Mode
            {proMode && (
              <span className="ml-1 text-violet-400">· up to 8 attempts</span>
            )}
          </button>
        ) : null}

        {/* error */}
        {phase === "error" && (
          <div className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-[0.7rem] text-red-400">
            {err}
          </div>
        )}

        {/* fuse button */}
        {phase !== "done" && (
          <button
            onClick={handleFuse}
            disabled={!canFuse}
            className={[
              "w-full py-4 rounded-2xl text-[0.85rem] font-mono tracking-[0.25em] uppercase transition-all duration-200",
              canFuse
                ? "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg shadow-violet-500/25 hover:scale-[1.02] active:scale-[0.98]"
                : busy
                  ? "bg-violet-600/40 text-white/60 cursor-not-allowed"
                  : "bg-white/5 text-white/20 border border-white/5 cursor-not-allowed",
            ].join(" ")}
          >
            {phase === "uploading" ? "Uploading…"
              : phase === "running" ? (job?.message ?? "Fusing…")
              : "⬡  Fuse"}
          </button>
        )}

        {/* progress bar */}
        {busy && job?.progress != null && (
          <div className="h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-600 to-fuchsia-500 rounded-full transition-all duration-700"
              style={{ width: `${job.progress}%` }}
            />
          </div>
        )}

        {/* live attempt log */}
        {phase === "running" && attempts.length > 0 && (
          <div className="rounded-2xl bg-white/[0.02] border border-white/5 p-4 flex flex-col gap-2">
            <span className="text-[0.6rem] uppercase tracking-widest text-muted2 mb-1">
              Optimization attempts
            </span>
            {attempts.map((att) => (
              <div
                key={att.n}
                className="flex items-center gap-2 text-[0.62rem] px-2 py-1.5 rounded-xl bg-white/[0.02] border border-white/5"
              >
                <span className="text-muted2 shrink-0">#{att.n}</span>
                <span className={[
                  "font-mono font-semibold tabular-nums",
                  att.chart_score >= 85 ? "text-yellow-400"
                    : att.chart_score >= 72 ? "text-violet-400"
                    : att.chart_score >= 58 ? "text-blue-400"
                    : "text-muted2",
                ].join(" ")}>
                  {att.chart_score}/100
                </span>
                <span className="text-muted2">{att.chart_grade}</span>
                {Object.keys(att.overrides).length > 0 && (
                  <span className="text-muted2/50 truncate ml-auto text-[0.58rem]">
                    {Object.entries(att.overrides)
                      .map(([k, v]) => `${k.replace(/_/g, " ")} ${(v as number) >= 0 ? "+" : ""}${(v as number).toFixed(2)}`)
                      .join(" · ")}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* output */}
        {phase === "done" && job?.output_url && (
          <div className="flex flex-col gap-4 animate-slide-up">

            <WavePlayer outputUrl={job.output_url} variants={job.variants ?? {}} />

            {/* scorecard */}
            {job.chart_score != null && (
              <ScoreCard
                chartScore={job.chart_score}
                chartGrade={job.chart_grade ?? "—"}
                mixScore={job.mix_score ?? job.chart_score}
                mixGrade={job.mix_grade ?? job.chart_grade ?? "—"}
                metadata={job.metadata}
                attempts={attempts.length > 1 ? attempts : undefined}
                seed={job.seed}
              />
            )}

            <button
              onClick={reset}
              className="w-full py-3 rounded-2xl border border-white/10 text-[0.7rem] font-mono tracking-[0.2em] uppercase text-white/40 hover:text-white hover:border-white/20 transition-all"
            >
              ↺  New Fusion
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
