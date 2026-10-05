#!/usr/bin/env python3
"""
grid_search_thresholds.py — Find optimal dual-filter thresholds for the SigLIP pipeline.

Searches over:
  - Static pixel diff threshold (MAD): mean absolute difference between consecutive
    grayscale frames below which a frame is considered static and discarded.
  - Semantic similarity threshold: cosine similarity above which a frame is
    considered a near-duplicate of the last kept frame and discarded.

Composite score = w_hit1 * Hit@1 + w_map * mAP + w_eff * (1 - retention_rate)
  The retention penalty rewards keeping fewer frames (faster search) without
  sacrificing retrieval quality.

Videos are downloaded once and reused across ALL threshold combinations,
so the expensive step runs only N times regardless of grid size.
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
import itertools
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal MAD + semantic-similarity thresholds"
    )
    parser.add_argument("--jsonl", default="highlight_train_release.jsonl",
                        help="QVHighlights JSONL file")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of dataset items to use (default: 10)")
    parser.add_argument("--output_dir", default="grid_search_results",
                        help="Directory for downloaded videos and result files")
    parser.add_argument("--download_timeout", type=int, default=60,
                        help="Per-video download timeout in seconds (0 = no limit)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K frames retrieved per query for scoring")
    # MAD grid
    parser.add_argument("--mad_values", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
                        help="MAD threshold values to search")
    # Semantic similarity grid
    parser.add_argument("--sem_values", nargs="+", type=float,
                        default=[0.85, 0.88, 0.90, 0.93, 0.95, 0.97],
                        help="Semantic similarity threshold values to search")
    # Composite score weights
    parser.add_argument("--w_hit1", type=float, default=0.40)
    parser.add_argument("--w_map",  type=float, default=0.40)
    parser.add_argument("--w_eff",  type=float, default=0.20,
                        help="Weight for frame retention efficiency (lower retention = better)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers  (same logic as evaluate_qvhighlights.py)
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
# In-process indexer  (bypasses .pkl cache so thresholds vary per combo)
# ---------------------------------------------------------------------------

def index_with_thresholds(
    video_path: str,
    mad_threshold: float,
    sem_threshold: float,
) -> Dict[str, Any]:
    """
    Index a video with the given thresholds.  Does NOT read/write a .pkl file.

    Returns dict with keys:
        embeddings         torch.Tensor (N, D)
        timestamps         list[float]
        skipped_static     int
        skipped_semantic   int
        total_candidates   int   (frames sampled at 1-fps rate)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            total_candidates += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Stage 1: static frame filter (MAD)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if diff.mean() < mad_threshold:
                    skipped_static += 1
                    prev_gray = gray
                    frame_id += 1
                    continue
            prev_gray = gray

            # Stage 2: semantic dedup against last kept frame
            emb = engine.encode_image(frame)
            if last_kept_emb is not None:
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

    emb_tensor = torch.stack(embeddings) if embeddings else torch.zeros(0, 512)
    return {
        "embeddings": emb_tensor,
        "timestamps": timestamps,
        "skipped_static": skipped_static,
        "skipped_semantic": skipped_semantic,
        "total_candidates": total_candidates,
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
    """Run a text query against the index and return retrieval metrics."""
    if embeddings.shape[0] == 0:
        return {"hit@1": 0.0, "hit@5": 0.0, "r@1_iou0.5": 0.0, "mrr": 0.0, "map": 0.0}

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
        results.append({"rank": rank, "timestamp": t, "score": score.item(),
                        "is_hit": is_hit, "max_iou": max_iou})

    hits = [r["is_hit"] for r in results]
    hit1 = 1.0 if hits and hits[0] else 0.0
    hit5 = 1.0 if any(hits[:5]) else 0.0
    r1_iou5 = 1.0 if results[0]["max_iou"] >= 0.5 else 0.0

    ranks_hit = [r["rank"] for r in results if r["is_hit"]]
    mrr = 1.0 / ranks_hit[0] if ranks_hit else 0.0

    num_rel, cum_p = 0, 0.0
    for idx_h, h in enumerate(hits, 1):
        if h:
            num_rel += 1
            cum_p += num_rel / idx_h
    ap = (cum_p / num_rel) if num_rel > 0 else 0.0

    return {"hit@1": hit1, "hit@5": hit5, "r@1_iou0.5": r1_iou5, "mrr": mrr, "map": ap}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.mad_values, args.sem_values))

    print("=" * 72)
    print("  SigLIP Dual-Filter Threshold Grid Search")
    print("=" * 72)
    print(f"  Dataset:       {args.jsonl}")
    print(f"  Samples:       {args.num_videos}")
    print(f"  MAD values:    {args.mad_values}")
    print(f"  Sem values:    {args.sem_values}")
    print(f"  Grid size:     {len(args.mad_values)} x {len(args.sem_values)} = {len(grid)} combos")
    print(f"  Score weights: Hit@1={args.w_hit1}  mAP={args.w_map}  Efficiency={args.w_eff}")
    print(f"  Output dir:    {out_dir.resolve()}")
    print("=" * 72)

    # 1. Load model once
    print("\n[1/4] Loading SigLIP model...")
    engine.load_model()
    print("      Done.\n")

    # 2. Load dataset
    print("[2/4] Reading dataset...")
    all_items = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    print(f"      {len(all_items)} total samples found.\n")

    # 3. Download videos once
    print(f"[3/4] Downloading up to {args.num_videos} videos (cached between runs)...")
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
        print(f"[OK] {video_path.name}")

    n = len(dataset)
    print(f"\n      {n} videos ready.\n")
    if n == 0:
        print("No videos available. Exiting.")
        return

    # 4. Grid search
    print(f"[4/4] Evaluating {len(grid)} threshold combinations across {n} videos...\n")
    header = f"  {'#':>3}  {'MAD':>5}  {'SEM':>5}  {'Hit@1':>6}  {'mAP':>6}  {'Ret%':>6}  {'Score':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results = []
    for combo_idx, (mad, sem) in enumerate(grid, 1):
        hit1_list, map_list, retention_list = [], [], []

        for item in dataset:
            video_path = item["video_path"]
            query = item["query"]
            rel_windows = item.get("relevant_windows", [])

            try:
                idx = index_with_thresholds(video_path, mad, sem)
            except Exception as e:
                print(f"\n    [Warning] Indexing failed ({video_path}): {e}")
                continue

            n_kept = len(idx["timestamps"])
            n_cand = idx["total_candidates"]
            retention = n_kept / n_cand if n_cand > 0 else 1.0

            metrics = search_and_score(
                idx["embeddings"], idx["timestamps"], query, rel_windows, args.top_k
            )
            hit1_list.append(metrics["hit@1"])
            map_list.append(metrics["map"])
            retention_list.append(retention)

        if not hit1_list:
            continue

        mean_hit1 = float(np.mean(hit1_list))
        mean_map  = float(np.mean(map_list))
        mean_ret  = float(np.mean(retention_list))
        composite = (
            args.w_hit1 * mean_hit1 +
            args.w_map  * mean_map  +
            args.w_eff  * (1.0 - mean_ret)
        )

        print(f"  {combo_idx:>3}  {mad:>5.2f}  {sem:>5.2f}  "
              f"{mean_hit1:>6.3f}  {mean_map:>6.3f}  "
              f"{mean_ret*100:>5.1f}%  {composite:>7.4f}")

        all_results.append({
            "mad_threshold": mad,
            "sem_threshold": sem,
            "hit@1": round(mean_hit1, 4),
            "mAP": round(mean_map, 4),
            "retention_rate": round(mean_ret, 4),
            "composite_score": round(composite, 4),
        })

    if not all_results:
        print("\nNo results produced.")
        return

    all_results.sort(key=lambda r: r["composite_score"], reverse=True)
    best = all_results[0]

    # Save JSON
    json_path = out_dir / "grid_search_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)

    # Save Markdown summary
    rows = "\n".join(
        f"| {r['mad_threshold']:.2f} | {r['sem_threshold']:.2f} | "
        f"{r['hit@1']:.3f} | {r['mAP']:.3f} | "
        f"{r['retention_rate']*100:.1f}% | **{r['composite_score']:.4f}** |"
        for r in all_results[:20]
    )
    summary_path = out_dir / "grid_search_summary.md"
    summary_md = f"""# SigLIP Threshold Grid Search Results

**Dataset:** `{args.jsonl}` ({n} videos evaluated)
**Grid:** MAD in {args.mad_values}, Semantic in {args.sem_values}
**Composite score** = {args.w_hit1}\u00d7Hit@1 + {args.w_map}\u00d7mAP + {args.w_eff}\u00d7(1\u2212retention)

## Best Configuration

| Parameter | Value |
|---|---|
| `STATIC_PIXEL_DIFF_THRESHOLD` | `{best['mad_threshold']}` |
| `SEMANTIC_SIM_THRESHOLD` | `{best['sem_threshold']}` |
| Hit@1 | {best['hit@1']} |
| mAP | {best['mAP']} |
| Frame retention | {best['retention_rate']*100:.1f}% |
| **Composite score** | **{best['composite_score']}** |

Apply in `backend/siglip_engine.py`:
```python
STATIC_PIXEL_DIFF_THRESHOLD = {best['mad_threshold']}
SEMANTIC_SIM_THRESHOLD      = {best['sem_threshold']}
```

## Top-20 Combinations

| MAD | Sem-Sim | Hit@1 | mAP | Retention | Score |
|---|---|---|---|---|---|
{rows}

*Full results: `grid_search_results.json`*
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\n" + "=" * 72)
    print("  GRID SEARCH COMPLETE")
    print("=" * 72)
    print(f"  Best MAD threshold:      {best['mad_threshold']}")
    print(f"  Best Semantic threshold: {best['sem_threshold']}")
    print(f"  Composite score:         {best['composite_score']}")
    print(f"  Hit@1={best['hit@1']}  mAP={best['mAP']}  Retention={best['retention_rate']*100:.1f}%")
    print()
    print("  Apply in backend/siglip_engine.py:")
    print(f"    STATIC_PIXEL_DIFF_THRESHOLD = {best['mad_threshold']}")
    print(f"    SEMANTIC_SIM_THRESHOLD      = {best['sem_threshold']}")
    print()
    print("  Reports:")
    print(f"    {summary_path.resolve()}")
    print(f"    {json_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
