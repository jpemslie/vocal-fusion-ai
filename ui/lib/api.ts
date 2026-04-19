export type JobStatus = "running" | "done" | "error";

export interface AttemptSummary {
  n: number;
  chart_score: number;
  chart_grade: string;
  overrides: Record<string, number>;
}

export interface SpectralMetadata {
  sibilance_ratio: number;
  delta_sub:       number;
  delta_bass:      number;
  delta_lo_mid:    number;
  delta_mid:       number;
  delta_hi_mid:    number;
  delta_presence:  number;
  delta_air:       number;
  lufs:            number;
  dynamic_range:   number;
  chart_details?:  Record<string, unknown>;
}

export interface Step {
  n:       number;
  status:  "pending" | "running" | "done" | "error";
  title:   string;
  elapsed: number | null;
  details: Record<string, unknown>;
}

export interface Job {
  status:      JobStatus;
  progress:    number;
  message:     string;
  name_a?:     string;
  name_b?:     string;
  seed?:       number;
  steps:       Step[];
  variants?:   { radio?: string; club?: string; intimate?: string };
  output_url?: string;
  share_url?:  string;

  // Scores
  score?:       number;
  chart_score?: number;
  chart_grade?: string;
  mix_score?:   number;
  mix_grade?:   string;

  // Spectral analysis
  metadata?: SpectralMetadata;

  // Refinement history
  attempts?: AttemptSummary[];
}

export async function startFuse(
  songA:       File,
  songB:       File,
  directVocal: boolean,
  seed:        number  = 42,
  autoRefine:  boolean = false,
  proMode:     boolean = true,
): Promise<string> {
  const fd = new FormData();
  fd.append("song_a", songA);
  fd.append("song_b", songB);
  fd.append("seed", String(seed));
  if (directVocal) fd.append("direct_vocal", "true");
  if (autoRefine)  fd.append("auto_refine",  "true");
  fd.append("pro_mode", proMode ? "true" : "false");

  const res = await fetch("/fuse", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? err.error ?? "Upload failed");
  }
  const data = await res.json();
  return data.job_id as string;
}

export async function getStatus(jobId: string): Promise<Job> {
  const res = await fetch(`/status/${jobId}`);
  if (!res.ok) throw new Error("Status check failed");
  return res.json();
}
