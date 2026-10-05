#!/usr/bin/env python3
"""
ablation_study.py — Ablation study for the SigLIP dual-filter pipeline.

Compares four conditions using the same N videos and queries:

  Condition A │ No filters      │ MAD disabled, Semantic disabled
  Condition B │ Static only     │ MAD active,   Semantic disabled
  Condition C │ Semantic only   │ MAD disabled, Semantic active
  Condition D │ Both (default)  │ MAD active,   Semantic active

"Disabled" means the threshold is set to its extreme (MAD=0 skips nothing,
Sem=1.0 skips nothing), so the code path is identical — only behaviour changes.

Metrics per condition:
  - Hit@1, Hit@5
  - R@1 (tIoU ≥ 0.5), R@5 (tIoU ≥ 0.5)
  - MRR, mAP
  - Retention rate  (fraction of 1-fps candidate frames kept)
  - Frames kept / skipped_static / skipped_semantic
  - Avg indexing time per video
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch

try:
    import imageio_ffmpeg
    raw_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = Path(raw_ffmpeg).parent
    standard_ffmpeg = ffmpeg_dir / "ffmpeg.exe"
    if not standard_ffmpeg.exists() and os.path.exists(raw_ffmpeg):
        import shutil
        shutil.copyfile(raw_ffmpeg, standard_ffmpeg)
    FFMPEG_EXE = str(standard_ffmpeg if standard_ffmpeg.exists() else raw_ffmpeg)
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    FFMPEG_EXE = "ffmpeg"

import yt_dlp

CURRENT_DIR = Path(__file__).resolve().parent
backend_dir = str((CURRENT_DIR / "backend").resolve())
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
# pyrefly: ignore [missing-import]
import siglip_engine as engine


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    name: str           # short label, e.g. "Both"
    label: str          # one-letter ID, e.g. "D"
    description: str    # human description
    mad: float          # MAD threshold  (0.0 = effectively disabled)
    sem: float          # Semantic threshold  (1.0 = effectively disabled)


# Sentinel values: MAD=0 → diff.mean() is never < 0 so nothing is skipped
#                  Sem=1.0 → cosine sim is always ≤ 1.0 so nothing is skipped
DEFAULT_CONDITIONS: List[Condition] = [
    Condition("No filters",    "A", "Baseline — all candidate frames kept",
              mad=0.0, sem=1.0),
    Condition("Static only",   "B", "Stage 1 (MAD) active, Stage 2 disabled",
              mad=2.0, sem=1.0),
    Condition("Semantic only", "C", "Stage 1 disabled, Stage 2 (cosine-sim) active",
              mad=0.0, sem=0.93),
    Condition("Both filters",  "D", "Default config — both stages active",
              mad=2.0, sem=0.93),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study: compare SigLIP dual-filter conditions"
    )
    parser.add_argument("--jsonl", default="highlight_train_release.jsonl",
                        help="QVHighlights JSONL file")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of dataset items (default: 10)")
    parser.add_argument("--output_dir", default="ablation_results",
                        help="Directory for videos and report files")
    parser.add_argument("--download_timeout", type=int, default=60,
                        help="Per-video download timeout in seconds (0 = no limit)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K frames retrieved per query")
    # Allow overriding the default MAD/sem values used for the active stages
    parser.add_argument("--mad", type=float, default=2.0,
                        help="MAD threshold to use when Stage 1 is active (default: 2.0)")
    parser.add_argument("--sem", type=float, default=0.93,
                        help="Semantic threshold to use when Stage 2 is active (default: 0.93)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def parse_vid(vid: str) -> Tuple[str, float, float]:
    parts = vid.split("_")
    return "_".join(parts[:-2]), float(parts[-2]), float(parts[-1])


def _run_yt_dlp_download(ydl_opts: dict, url: str) -> None:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_trimmed_video(
    youtube_id: str, start_time: float, end_time: float,
    output_path: Path, timeout: int = 60,
) -> bool:
    """Return True on success, False if unavailable. Raises TimeoutError on stall."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_template = str(output_path.with_suffix(".tmp.%(ext)s"))
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        # Use bestvideo+bestaudio so FFmpegFD (the only downloader that supports
        # download_ranges / partial cuts) is always selected. Avoid ext=mp4 / ext=m4a
        # constraints that can push yt-dlp toward native/DASH protocols incompatible
        # with partial download.
        "format": "bestvideo[height<=480]+bestaudio/bestvideo[height<=480]/best[height<=480]/best",
        "merge_output_format": "mp4",
        "outtmpl": temp_template,
        "ffmpeg_location": os.path.dirname(FFMPEG_EXE),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": min(timeout, 30) if timeout > 0 else 30,
    }

    def _cleanup():
        for f in output_path.parent.glob(output_path.stem + ".tmp.*"):
            try:
                f.unlink()
            except Exception:
                pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_yt_dlp_download, ydl_opts, url)
            try:
                fut.result(timeout=timeout if timeout > 0 else None)
            except concurrent.futures.TimeoutError:
                _cleanup()
                raise TimeoutError(
                    f"Download of {youtube_id} exceeded {timeout}s — "
                    "check your internet connection and re-run."
                )
        candidates = list(output_path.parent.glob(output_path.stem + ".tmp.*"))
        if candidates:
            downloaded = candidates[0]
            if output_path.exists():
                output_path.unlink()
            downloaded.rename(output_path)
            return True
    except TimeoutError:
        raise
    except Exception as e:
        print(f"    [Warning] Download failed for {youtube_id}: {e}")
        _cleanup()
        return False
    return False


def download_with_retry(
    youtube_id: str,
    start_time: float,
    end_time: float,
    output_path: "Path",
    timeout: int = 60,
    initial_wait: int = 30,
    max_wait: int = 300,
) -> bool:
    """
    Wrapper around download_trimmed_video that retries indefinitely on timeout.

    On TimeoutError (network stall) it waits `wait` seconds (exponential backoff,
    capped at max_wait) then tries again. Returns only when:
      - True  : download succeeded
      - False : video is genuinely unavailable / private (non-retryable error)
    """
    wait = initial_wait
    attempt = 0
    while True:
        try:
            return download_trimmed_video(
                youtube_id, start_time, end_time, output_path, timeout=timeout
            )
        except TimeoutError:
            attempt += 1
            print(
                f"\n  [Timeout] Download stalled (attempt {attempt}). "
                f"Waiting {wait}s before retry — internet will be re-checked automatically..."
            )
            time.sleep(wait)
            wait = min(wait * 2, max_wait)
            print(f"  [Retry] Re-attempting download of {youtube_id}...")


# ---------------------------------------------------------------------------
# In-process indexer (no .pkl — conditions differ per run)
# ---------------------------------------------------------------------------

def index_with_thresholds(
    video_path: str,
    mad_threshold: float,
    sem_threshold: float,
) -> Dict[str, Any]:
    """
    Index a video with explicit thresholds, bypassing any .pkl cache.

    MAD = 0.0  → Stage 1 disabled (diff.mean() is always >= 0)
    Sem = 1.0  → Stage 2 disabled (cosine sim is always <= 1.0)

    Returns:
        embeddings        torch.Tensor (N, D)
        timestamps        list[float]
        skipped_static    int
        skipped_semantic  int
        total_candidates  int
        elapsed_sec       float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps), 1)

    embeddings: List[torch.Tensor] = []
    timestamps: List[float] = []
    last_kept_emb = None
    prev_gray = None
    frame_id = 0
    skipped_static = 0
    skipped_semantic = 0
    total_candidates = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            total_candidates += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Stage 1: static frame filter
            if prev_gray is not None and mad_threshold > 0.0:
                diff = cv2.absdiff(prev_gray, gray)
                if diff.mean() < mad_threshold:
                    skipped_static += 1
                    prev_gray = gray
                    frame_id += 1
                    continue
            prev_gray = gray

            # Stage 2: semantic dedup
            emb = engine.encode_image(frame)
            if last_kept_emb is not None and sem_threshold < 1.0:
                sim = (emb @ last_kept_emb).item()
                if sim > sem_threshold:
                    skipped_semantic += 1
                    frame_id += 1
                    continue

            embeddings.append(emb)
            timestamps.append(frame_id / fps)
            last_kept_emb = emb

        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    emb_tensor = torch.stack(embeddings) if embeddings else torch.zeros(0, 512)
    return {
        "embeddings": emb_tensor,
        "timestamps": timestamps,
        "skipped_static": skipped_static,
        "skipped_semantic": skipped_semantic,
        "total_candidates": total_candidates,
        "elapsed_sec": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def temporal_iou(w1: Tuple[float, float], w2: Tuple[float, float]) -> float:
    inter = max(0.0, min(w1[1], w2[1]) - max(w1[0], w2[0]))
    union = (w1[1] - w1[0]) + (w2[1] - w2[0]) - inter
    return inter / union if union > 0 else 0.0


def search_and_score(
    embeddings: torch.Tensor,
    timestamps: List[float],
    query: str,
    relevant_windows: List[List[float]],
    top_k: int,
) -> Dict[str, float]:
    if embeddings.shape[0] == 0:
        return {"hit@1": 0.0, "hit@5": 0.0, "r@1_iou0.5": 0.0, "r@5_iou0.5": 0.0,
                "mrr": 0.0, "map": 0.0}

    text_emb = engine.encode_text(query)
    scores = embeddings @ text_emb
    k = min(top_k, len(scores))
    values, indices = torch.topk(scores, k=k)

    results = []
    for rank, (score, idx) in enumerate(zip(values, indices), 1):
        i = idx.item()
        t = timestamps[i]
        pred_win = (max(0.0, t - 1.0), t + 1.0)
        max_iou = max(
            (temporal_iou(pred_win, (float(w[0]), float(w[1]))) for w in relevant_windows),
            default=0.0,
        )
        is_hit = any(w[0] <= t <= w[1] for w in relevant_windows)
        results.append({"rank": rank, "is_hit": is_hit, "max_iou": max_iou})

    hits = [r["is_hit"] for r in results]
    hit1 = 1.0 if hits and hits[0] else 0.0
    hit5 = 1.0 if any(hits[:5]) else 0.0
    r1_iou5 = 1.0 if results and results[0]["max_iou"] >= 0.5 else 0.0
    r5_iou5 = 1.0 if any(r["max_iou"] >= 0.5 for r in results[:5]) else 0.0

    ranks_hit = [r["rank"] for r in results if r["is_hit"]]
    mrr = 1.0 / ranks_hit[0] if ranks_hit else 0.0

    num_rel, cum_p = 0, 0.0
    for i_h, h in enumerate(hits, 1):
        if h:
            num_rel += 1
            cum_p += num_rel / i_h
    ap = (cum_p / num_rel) if num_rel > 0 else 0.0

    return {"hit@1": hit1, "hit@5": hit5, "r@1_iou0.5": r1_iou5, "r@5_iou0.5": r5_iou5,
            "mrr": mrr, "map": ap}


# ---------------------------------------------------------------------------
# Run one condition over all dataset videos
# ---------------------------------------------------------------------------

def run_condition(
    condition: Condition,
    dataset: List[Dict],
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate one ablation condition and aggregate metrics."""
    hit1_l, hit5_l, r1_l, r5_l, mrr_l, map_l = [], [], [], [], [], []
    ret_l, skip_s_l, skip_sem_l, kept_l, elapsed_l = [], [], [], [], []

    for item in dataset:
        video_path = item["video_path"]
        query = item["query"]
        rel_windows = item.get("relevant_windows", [])

        try:
            idx = index_with_thresholds(video_path, condition.mad, condition.sem)
        except Exception as e:
            print(f"    [Warning] Indexing failed ({Path(video_path).name}): {e}")
            continue

        n_kept = len(idx["timestamps"])
        n_cand = idx["total_candidates"]
        retention = n_kept / n_cand if n_cand > 0 else 1.0

        metrics = search_and_score(
            idx["embeddings"], idx["timestamps"], query, rel_windows, top_k
        )

        hit1_l.append(metrics["hit@1"])
        hit5_l.append(metrics["hit@5"])
        r1_l.append(metrics["r@1_iou0.5"])
        r5_l.append(metrics["r@5_iou0.5"])
        mrr_l.append(metrics["mrr"])
        map_l.append(metrics["map"])
        ret_l.append(retention)
        skip_s_l.append(idx["skipped_static"])
        skip_sem_l.append(idx["skipped_semantic"])
        kept_l.append(n_kept)
        elapsed_l.append(idx["elapsed_sec"])

    def _m(lst): return float(np.mean(lst)) if lst else 0.0

    return {
        "condition": condition.label,
        "name": condition.name,
        "description": condition.description,
        "mad_threshold": condition.mad,
        "sem_threshold": condition.sem,
        "n_evaluated": len(hit1_l),
        # Quality metrics
        "hit@1": round(_m(hit1_l), 4),
        "hit@5": round(_m(hit5_l), 4),
        "r@1_iou0.5": round(_m(r1_l), 4),
        "r@5_iou0.5": round(_m(r5_l), 4),
        "mrr": round(_m(mrr_l), 4),
        "mAP": round(_m(map_l), 4),
        # Efficiency metrics
        "avg_retention_rate": round(_m(ret_l), 4),
        "avg_frames_kept": round(_m(kept_l), 1),
        "avg_skipped_static": round(_m(skip_s_l), 1),
        "avg_skipped_semantic": round(_m(skip_sem_l), 1),
        "avg_index_time_sec": round(_m(elapsed_l), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Patch condition thresholds from CLI overrides
    conditions = [
        Condition("No filters",    "A", "Baseline — all candidate frames kept",
                  mad=0.0, sem=1.0),
        Condition("Static only",   "B", f"Stage 1 (MAD={args.mad}) active, Stage 2 disabled",
                  mad=args.mad, sem=1.0),
        Condition("Semantic only", "C", f"Stage 1 disabled, Stage 2 (sim>{args.sem}) active",
                  mad=0.0, sem=args.sem),
        Condition("Both filters",  "D", f"Both stages active (MAD={args.mad}, sim>{args.sem})",
                  mad=args.mad, sem=args.sem),
    ]

    print("=" * 72)
    print("  SigLIP Dual-Filter Ablation Study")
    print("=" * 72)
    print(f"  Dataset:      {args.jsonl}")
    print(f"  Samples:      {args.num_videos}")
    print(f"  MAD value:    {args.mad}")
    print(f"  Sem value:    {args.sem}")
    print(f"  Conditions:   {len(conditions)}")
    print(f"  Output dir:   {out_dir.resolve()}")
    print()
    for c in conditions:
        print(f"  [{c.label}] {c.name:<18} — {c.description}")
    print("=" * 72)

    # 1. Load model once
    print("\n[1/4] Loading SigLIP model...")
    engine.load_model()
    print("      Done.\n")

    # 2. Read dataset
    print("[2/4] Reading dataset...")
    all_items = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    print(f"      {len(all_items)} total samples.\n")

    # 3. Download videos once
    print(f"[3/4] Downloading up to {args.num_videos} videos (reused across all conditions)...")
    dataset: List[Dict] = []
    for item in all_items:
        if len(dataset) >= args.num_videos:
            break
        vid = item["vid"]
        youtube_id, start_s, end_s = parse_vid(vid)
        video_path = videos_dir / f"{youtube_id}_{int(start_s)}_{int(end_s)}.mp4"
        # Download — retries automatically on timeout until internet returns
        print(f"  [{len(dataset)+1}/{args.num_videos}] {vid}", end="  ")
        ok = download_with_retry(
            youtube_id, start_s, end_s, video_path,
            timeout=args.download_timeout,
        )
        if not ok:
            print("[Skipped — unavailable]")
            continue
        dataset.append({**item, "video_path": str(video_path)})
        print(f"[OK]")

    n = len(dataset)
    print(f"\n      {n} videos ready.\n")
    if n == 0:
        print("No videos available. Exiting.")
        return

    # 4. Run ablation
    print("[4/4] Running ablation conditions...\n")
    all_cond_results = []
    for cond in conditions:
        print(f"  ── [{cond.label}] {cond.name} ──────────────────────────────────")
        print(f"       MAD={cond.mad}  SEM={cond.sem}")
        t_wall = time.time()
        result = run_condition(cond, dataset, args.top_k)
        wall = time.time() - t_wall
        result["wall_time_sec"] = round(wall, 1)
        all_cond_results.append(result)
        print(f"       Hit@1={result['hit@1']:.3f}  Hit@5={result['hit@5']:.3f}  "
              f"mAP={result['mAP']:.3f}  MRR={result['mrr']:.3f}")
        print(f"       Retention={result['avg_retention_rate']*100:.1f}%  "
              f"Kept={result['avg_frames_kept']:.0f}  "
              f"SkipStatic={result['avg_skipped_static']:.0f}  "
              f"SkipSem={result['avg_skipped_semantic']:.0f}  "
              f"IndexTime={result['avg_index_time_sec']:.2f}s\n")

    # ── Derive contribution deltas relative to "No filters" baseline ──────────
    baseline = next(r for r in all_cond_results if r["condition"] == "A")
    for r in all_cond_results:
        r["delta_hit1_vs_baseline"] = round(r["hit@1"] - baseline["hit@1"], 4)
        r["delta_map_vs_baseline"]  = round(r["mAP"]   - baseline["mAP"],   4)
        r["speedup_vs_baseline"]    = (
            round(baseline["avg_index_time_sec"] / r["avg_index_time_sec"], 2)
            if r["avg_index_time_sec"] > 0 else 1.0
        )

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_cond_results}, f, indent=2)

    # ── Build Markdown report ──────────────────────────────────────────────────
    def pct(v): return f"{v*100:.1f}%"
    def pm(v):  return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

    quality_rows = "\n".join(
        f"| [{r['condition']}] {r['name']:<18} | {r['hit@1']:.3f} | {r['hit@5']:.3f} | "
        f"{r['r@1_iou0.5']:.3f} | {r['r@5_iou0.5']:.3f} | {r['mrr']:.3f} | {r['mAP']:.3f} | "
        f"{pm(r['delta_hit1_vs_baseline'])} | {pm(r['delta_map_vs_baseline'])} |"
        for r in all_cond_results
    )

    efficiency_rows = "\n".join(
        f"| [{r['condition']}] {r['name']:<18} | {pct(r['avg_retention_rate'])} | "
        f"{r['avg_frames_kept']:.0f} | {r['avg_skipped_static']:.0f} | "
        f"{r['avg_skipped_semantic']:.0f} | {r['avg_index_time_sec']:.3f}s | "
        f"{r['speedup_vs_baseline']:.2f}× |"
        for r in all_cond_results
    )

    # Compute filter-specific contribution
    static_only  = next(r for r in all_cond_results if r["condition"] == "B")
    sem_only     = next(r for r in all_cond_results if r["condition"] == "C")
    both         = next(r for r in all_cond_results if r["condition"] == "D")

    static_frames_saved  = baseline["avg_frames_kept"] - static_only["avg_frames_kept"]
    sem_frames_saved     = baseline["avg_frames_kept"] - sem_only["avg_frames_kept"]
    combined_frames_saved = baseline["avg_frames_kept"] - both["avg_frames_kept"]

    summary_path = out_dir / "ablation_report.md"
    report = f"""# SigLIP Dual-Filter Ablation Study

**Dataset:** `{args.jsonl}` — {n} videos, {args.num_videos} requested  
**MAD threshold (when active):** `{args.mad}`  
**Semantic threshold (when active):** `{args.sem}`  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Conditions

| ID | Condition | Stage 1 (MAD) | Stage 2 (Sem-Sim) |
|---|---|---|---|
| A | No filters | disabled | disabled |
| B | Static only | `MAD < {args.mad}` | disabled |
| C | Semantic only | disabled | `sim > {args.sem}` |
| D | Both (default) | `MAD < {args.mad}` | `sim > {args.sem}` |

---

## Retrieval Quality

| Condition | Hit@1 | Hit@5 | R@1 IoU≥0.5 | R@5 IoU≥0.5 | MRR | mAP | ΔHit@1 | ΔmAP |
|---|---|---|---|---|---|---|---|---|
{quality_rows}

*Δ values are relative to the No-filters baseline [A].*

---

## Indexing Efficiency

| Condition | Retention | Frames Kept | SkipStatic | SkipSem | Avg Index Time | Speedup |
|---|---|---|---|---|---|---|
{efficiency_rows}

*Speedup is relative to baseline [A] indexing time.*

---

## Filter Contribution Analysis

| Filter | Frames eliminated (avg) | Retention reduction | Index speedup |
|---|---|---|---|
| Stage 1 — Static (MAD) | {static_frames_saved:.1f} frames | {(1-static_only['avg_retention_rate'])*100:.1f}% → {(1-baseline['avg_retention_rate'])*100:.1f}% | {static_only['speedup_vs_baseline']:.2f}× |
| Stage 2 — Semantic (sim) | {sem_frames_saved:.1f} frames | {(1-sem_only['avg_retention_rate'])*100:.1f}% → {(1-baseline['avg_retention_rate'])*100:.1f}% | {sem_only['speedup_vs_baseline']:.2f}× |
| Both combined | {combined_frames_saved:.1f} frames | {(1-both['avg_retention_rate'])*100:.1f}% → {(1-baseline['avg_retention_rate'])*100:.1f}% | {both['speedup_vs_baseline']:.2f}× |

### Quality trade-off (vs baseline)

| Filter | ΔHit@1 | ΔmAP | Verdict |
|---|---|---|---|
| Stage 1 — Static | {pm(static_only['delta_hit1_vs_baseline'])} | {pm(static_only['delta_map_vs_baseline'])} | {"✅ No quality loss" if static_only['delta_hit1_vs_baseline'] >= -0.01 else "⚠️ Some quality loss"} |
| Stage 2 — Semantic | {pm(sem_only['delta_hit1_vs_baseline'])} | {pm(sem_only['delta_map_vs_baseline'])} | {"✅ No quality loss" if sem_only['delta_hit1_vs_baseline'] >= -0.01 else "⚠️ Some quality loss"} |
| Both combined | {pm(both['delta_hit1_vs_baseline'])} | {pm(both['delta_map_vs_baseline'])} | {"✅ No quality loss" if both['delta_hit1_vs_baseline'] >= -0.01 else "⚠️ Some quality loss"} |

---

*Full per-condition data: `ablation_results.json`*
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Final console summary ──────────────────────────────────────────────────
    print("=" * 72)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 72)
    # ASCII table
    header = f"  {'Cond':<22} {'Hit@1':>6} {'mAP':>6} {'Ret%':>6} {'Time':>8} {'Speedup':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_cond_results:
        marker = " ◄ baseline" if r["condition"] == "A" else ""
        print(f"  [{r['condition']}] {r['name']:<18} "
              f"{r['hit@1']:>6.3f} {r['mAP']:>6.3f} "
              f"{r['avg_retention_rate']*100:>5.1f}% "
              f"{r['avg_index_time_sec']:>7.3f}s "
              f"{r['speedup_vs_baseline']:>6.2f}×"
              f"{marker}")
    print()
    print("  Reports:")
    print(f"    {summary_path.resolve()}")
    print(f"    {json_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
