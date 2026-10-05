#!/usr/bin/env python3
"""
evaluate_qvhighlights.py — Benchmark SigLIP pipeline on QVHighlights dataset.

Features:
- Downloads YouTube videos and trims exact segment defined by QVHighlights vid ({youtube_id}_{start}_{end}).
- Indexes frames using SigLIP with dual-stage static/semantic deduplication.
- Executes queries, retrieves top-K matches, and maps frames to 2-second clip windows.
- Calculates standard research metrics:
    * Recall@1, Recall@5, Recall@10 (IoU >= 0.5 and IoU >= 0.7)
    * Mean Reciprocal Rank (MRR)
    * Mean Average Precision (mAP)
    * Hit@1, Hit@5 (whether top retrieved frame falls directly inside ground truth relevant windows)
    * Saliency correlation (Spearman rank correlation with human saliency annotations)
- Saves full evaluation artifacts:
    * Detailed per-video/query metrics CSV
    * Research paper-ready summary report (Markdown & JSON)
    * Best retrieved frames saved as images for visual inspection
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
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

# Add backend directory to sys.path to load SigLIP engine
CURRENT_DIR = Path(__file__).resolve().parent
backend_dir = str((CURRENT_DIR / "backend").resolve())
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
# pyrefly: ignore [missing-import]
import siglip_engine as engine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SigLIP semantic video search on QVHighlights dataset"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="highlight_train_release.jsonl",
        help="Path to QVHighlights jsonl file (default: highlight_train_release.jsonl)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=10,
        help="Maximum number of video/query items to evaluate (default: 5, set -1 for all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save downloaded videos, indices, and evaluation reports",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K frames to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--skip_download_errors",
        action="store_true",
        default=True,
        help="Skip videos that fail to download (e.g., unavailable on YouTube)",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        default=True,
        help="Save top retrieved frame images for visual inspection in report",
    )
    parser.add_argument(
        "--download_timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each video download (default: 60). Set to 0 to disable.",
    )
    return parser.parse_args()


def parse_vid(vid: str) -> Tuple[str, float, float]:
    """Parse vid formatted as {youtube_id}_{start_time}_{end_time}"""
    parts = vid.split("_")
    youtube_id = "_".join(parts[:-2])
    start_time = float(parts[-2])
    end_time = float(parts[-1])
    return youtube_id, start_time, end_time


def _run_yt_dlp_download(ydl_opts: dict, url: str) -> None:
    """Worker function that runs yt-dlp inside a thread (needed for timeout support)."""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_trimmed_video(
    youtube_id: str,
    start_time: float,
    end_time: float,
    output_path: Path,
    timeout: int = 60,
) -> bool:
    """Download trimmed YouTube video using yt-dlp and ffmpeg.

    Args:
        youtube_id: YouTube video identifier.
        start_time: Clip start in seconds.
        end_time: Clip end in seconds.
        output_path: Where to write the final .mp4 file.
        timeout: Max seconds to wait for the download; 0 = no limit.

    Returns:
        True on success, False if the video is genuinely unavailable/private.

    Raises:
        TimeoutError: If the download stalls longer than `timeout` seconds.
                      The caller should halt — not skip — so the video can be
                      retried on the next run without inflating the sample count.
    """
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
        # Socket-level timeouts so yt-dlp itself does not stall on a dead connection
        "socket_timeout": min(timeout, 30) if timeout > 0 else 30,
    }

    def _cleanup_temp() -> None:
        for f in output_path.parent.glob(output_path.stem + ".tmp.*"):
            try:
                f.unlink()
            except Exception:
                pass

    try:
        effective_timeout = timeout if timeout > 0 else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_yt_dlp_download, ydl_opts, url)
            try:
                future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError:
                _cleanup_temp()
                raise TimeoutError(
                    f"Download of {youtube_id} exceeded {timeout}s — "
                    f"possible network issue. Re-run after restoring connectivity."
                )

        # Find the actual downloaded file and rename to output_path
        candidates = list(output_path.parent.glob(output_path.stem + ".tmp.*"))
        if candidates:
            downloaded = candidates[0]
            if output_path.exists():
                output_path.unlink()
            downloaded.rename(output_path)
            return True
    except TimeoutError:
        raise  # propagate — do not swallow
    except Exception as e:
        print(f"  [Warning] Failed to download {youtube_id} ({start_time}-{end_time}): {e}")
        _cleanup_temp()
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
                f"  [Timeout] Download stalled (attempt {attempt}). "
                f"Waiting {wait}s before retry — internet will be re-checked automatically..."
            )
            time.sleep(wait)
            wait = min(wait * 2, max_wait)
            print(f"  [Retry] Re-attempting download of {youtube_id}...")


def calculate_temporal_iou(window1: Tuple[float, float], window2: Tuple[float, float]) -> float:
    """Calculate temporal Intersection over Union (tIoU) between two time spans."""
    start1, end1 = window1
    start2, end2 = window2
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter = max(0.0, inter_end - inter_start)
    union = (end1 - start1) + (end2 - start2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_metrics(
    retrieved_results: List[Dict[str, Any]],
    relevant_windows: List[List[float]],
    relevant_clip_ids: List[int],
    saliency_scores: List[List[int]],
    clip_length: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute information retrieval and temporal localization metrics:
    - Hit@1, Hit@5, Hit@10: retrieved timestamp falls inside any ground truth relevant window.
    - Recall@1, Recall@5, Recall@10 at IoU 0.5 & 0.7: temporal 2s window around frame matches GT window.
    - MRR: Mean Reciprocal Rank of first relevant frame.
    - Average Precision (AP): Area under PR curve across top retrievals.
    - Saliency rank correlation (Spearman rho): correlation between model scores and human saliency.
    """
    ranks = []
    hits = []
    ap_hits = []

    # Map ground truth clip_ids to average human saliency score (out of 4.0)
    clip_gt_saliency = {}
    if relevant_clip_ids and saliency_scores:
        for clip_id, scores in zip(relevant_clip_ids, saliency_scores):
            clip_gt_saliency[clip_id] = float(np.mean(scores))

    model_scores_for_corr = []
    gt_scores_for_corr = []

    for item in retrieved_results:
        t = item["timestamp"]
        rank = item["rank"]
        score = item["score"]

        # Check if timestamp falls inside any relevant window
        is_hit = any(w[0] <= t <= w[1] for w in relevant_windows)
        hits.append(is_hit)
        if is_hit:
            ranks.append(rank)

        # 2-second predicted clip window [t - 1.0, t + 1.0] bounded at 0
        pred_window = (max(0.0, t - clip_length / 2.0), t + clip_length / 2.0)
        max_iou = 0.0
        for w in relevant_windows:
            iou = calculate_temporal_iou(pred_window, (float(w[0]), float(w[1])))
            if iou > max_iou:
                max_iou = iou
        item["max_iou"] = max_iou

        # Associate with 2-second clip id
        clip_id = int(t // clip_length)
        if clip_id in clip_gt_saliency:
            model_scores_for_corr.append(score)
            gt_scores_for_corr.append(clip_gt_saliency[clip_id])

    # Hit@K
    hit_1 = 1.0 if len(hits) > 0 and hits[0] else 0.0
    hit_5 = 1.0 if any(hits[:5]) else 0.0
    hit_10 = 1.0 if any(hits[:10]) else 0.0

    # Recall@K at IoU thresholds
    r1_iou5 = 1.0 if len(retrieved_results) > 0 and retrieved_results[0]["max_iou"] >= 0.5 else 0.0
    r5_iou5 = 1.0 if any(r["max_iou"] >= 0.5 for r in retrieved_results[:5]) else 0.0
    r10_iou5 = 1.0 if any(r["max_iou"] >= 0.5 for r in retrieved_results[:10]) else 0.0

    r1_iou7 = 1.0 if len(retrieved_results) > 0 and retrieved_results[0]["max_iou"] >= 0.7 else 0.0
    r5_iou7 = 1.0 if any(r["max_iou"] >= 0.7 for r in retrieved_results[:5]) else 0.0
    r10_iou7 = 1.0 if any(r["max_iou"] >= 0.7 for r in retrieved_results[:10]) else 0.0

    # Reciprocal Rank
    rr = 1.0 / ranks[0] if ranks else 0.0

    # Average Precision (AP@K): standard information retrieval precision across relevant ranks
    num_relevant = 0
    cum_precision = 0.0
    for idx, h in enumerate(hits, start=1):
        if h:
            num_relevant += 1
            cum_precision += num_relevant / idx
    ap = (cum_precision / num_relevant) if num_relevant > 0 else 0.0

    # Spearman rank correlation
    saliency_spearman = None
    if len(model_scores_for_corr) >= 3:
        try:
            # Simple Spearman rank correlation implementation without scipy dependency
            def get_ranks(vals):
                order = np.argsort(vals)
                r = np.empty_like(order, dtype=float)
                r[order] = np.arange(len(vals))
                return r

            r_m = get_ranks(model_scores_for_corr)
            r_g = get_ranks(gt_scores_for_corr)
            d = r_m - r_g
            n = len(model_scores_for_corr)
            rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
            saliency_spearman = float(rho)
        except Exception:
            saliency_spearman = None

    return {
        "hit@1": hit_1,
        "hit@5": hit_5,
        "hit@10": hit_10,
        "r@1_iou0.5": r1_iou5,
        "r@5_iou0.5": r5_iou5,
        "r@10_iou0.5": r10_iou5,
        "r@1_iou0.7": r1_iou7,
        "r@5_iou0.7": r5_iou7,
        "r@10_iou0.7": r10_iou7,
        "reciprocal_rank": rr,
        "average_precision": ap,
        "saliency_correlation": saliency_spearman,
    }


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    videos_dir = out_dir / "videos"
    frames_dir = out_dir / "retrieved_frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Camiq SigLIP Engine Benchmark on QVHighlights Dataset")
    print("=" * 70)
    print(f"Dataset path:      {args.jsonl}")
    print(f"Max items:         {args.max_videos if args.max_videos > 0 else 'All'}")
    print(f"Top-K frame eval:  {args.top_k}")
    print(f"Output directory:  {out_dir.resolve()}")
    print("=" * 70)

    # 1. Load model once
    print("\n[Step 1/3] Initializing SigLIP model...")
    engine.load_model()
    print("Model initialized successfully.\n")

    # 2. Read dataset entries
    print("[Step 2/3] Loading dataset...")
    items = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    print(f"Loaded {len(items)} samples from dataset.\n")

    # 3. Evaluate each sample
    print("[Step 3/3] Running evaluation loop...")
    results_records = []
    start_eval_time = time.time()

    for idx, item in enumerate(items, start=1):
        if args.max_videos > 0 and len(results_records) >= args.max_videos:
            print(f"\n[Info] Successfully evaluated target of {args.max_videos} samples. Stopping.")
            break

        qid = item["qid"]
        query = item["query"]
        vid = item["vid"]
        duration = item.get("duration", 0)
        rel_windows = item.get("relevant_windows", [])
        rel_clip_ids = item.get("relevant_clip_ids", [])
        saliency_scores = item.get("saliency_scores", [])

        target_display = args.max_videos if args.max_videos > 0 else len(items)
        current_num = len(results_records) + 1
        print(f"\n[Evaluating {current_num}/{target_display}] QID {qid} | VID: {vid}")
        print(f"  Query: \"{query}\"")
        print(f"  Ground Truth Windows: {rel_windows}")

        youtube_id, start_s, end_s = parse_vid(vid)
        video_filename = f"{youtube_id}_{int(start_s)}_{int(end_s)}.mp4"
        video_path = videos_dir / video_filename

        # Download if needed — retries automatically on timeout until internet returns
        print(f"  Downloading video ({start_s}s - {end_s}s)... [timeout: {args.download_timeout}s]")
        success = download_with_retry(
            youtube_id, start_s, end_s, video_path, timeout=args.download_timeout
        )

        if not success:
            # Genuinely unavailable/private — safe to skip
            if args.skip_download_errors:
                print(f"  [Skipping] Video is unavailable on YouTube. Moving to next candidate...")
                continue
            else:
                raise RuntimeError(f"Could not download {youtube_id}")

        # Index video using SigLIP engine
        print(f"  Indexing video with SigLIP pipeline...")
        t0 = time.time()
        engine.build_index(str(video_path))
        index_duration = time.time() - t0
        stats = engine.get_status().get("stats", {})

        # Search query
        print(f"  Retrieving top-{args.top_k} frames for query...")
        search_res = engine.search(query, top_k=args.top_k)

        # Save top frame preview if enabled
        top_frame_saved = None
        if args.save_frames and len(search_res) > 0:
            top_frame_idx = search_res[0]["frame_index"]
            top_frame_bytes = engine.get_frame(top_frame_idx)
            frame_filename = f"qid_{qid}_rank1_{search_res[0]['timestamp_str'].replace(':', '-')}.jpg"
            frame_filepath = frames_dir / frame_filename
            with open(frame_filepath, "wb") as img_f:
                img_f.write(top_frame_bytes)
            top_frame_saved = str(frame_filepath.name)

        # Compute benchmark metrics
        metrics = compute_metrics(
            retrieved_results=search_res,
            relevant_windows=rel_windows,
            relevant_clip_ids=rel_clip_ids,
            saliency_scores=saliency_scores,
        )

        record = {
            "qid": qid,
            "vid": vid,
            "query": query,
            "duration": duration,
            "video_path": str(video_path),
            "index_time_sec": round(index_duration, 2),
            "indexed_frames": stats.get("total_frames", 0),
            "skipped_static": stats.get("skipped_static", 0),
            "skipped_semantic": stats.get("skipped_semantic", 0),
            "hit@1": metrics["hit@1"],
            "hit@5": metrics["hit@5"],
            "hit@10": metrics["hit@10"],
            "r@1_iou0.5": metrics["r@1_iou0.5"],
            "r@5_iou0.5": metrics["r@5_iou0.5"],
            "r@1_iou0.7": metrics["r@1_iou0.7"],
            "r@5_iou0.7": metrics["r@5_iou0.7"],
            "reciprocal_rank": round(metrics["reciprocal_rank"], 4),
            "average_precision": round(metrics["average_precision"], 4),
            "saliency_spearman": round(metrics["saliency_correlation"], 4) if metrics["saliency_correlation"] is not None else "N/A",
            "top1_timestamp": search_res[0]["timestamp_str"] if search_res else "N/A",
            "top1_score": search_res[0]["score"] if search_res else 0.0,
            "top_frame_file": top_frame_saved or "N/A",
            "gt_windows": str(rel_windows),
        }
        results_records.append(record)

        print(f"  Results: Hit@1: {metrics['hit@1']} | Hit@5: {metrics['hit@5']} | MRR: {metrics['reciprocal_rank']:.2f} | AP: {metrics['average_precision']:.2f}")

    total_eval_time = round(time.time() - start_eval_time, 2)
    n = len(results_records)

    if n == 0:
        print("\nNo items were successfully processed.")
        return

    # Calculate overall aggregates for research paper
    mean_hit1 = np.mean([r["hit@1"] for r in results_records])
    mean_hit5 = np.mean([r["hit@5"] for r in results_records])
    mean_hit10 = np.mean([r["hit@10"] for r in results_records])
    mean_r1_iou5 = np.mean([r["r@1_iou0.5"] for r in results_records])
    mean_r5_iou5 = np.mean([r["r@5_iou0.5"] for r in results_records])
    mean_r1_iou7 = np.mean([r["r@1_iou0.7"] for r in results_records])
    mean_r5_iou7 = np.mean([r["r@5_iou0.7"] for r in results_records])
    mean_mrr = np.mean([r["reciprocal_rank"] for r in results_records])
    mean_map = np.mean([r["average_precision"] for r in results_records])

    spearmans = [r["saliency_spearman"] for r in results_records if r["saliency_spearman"] != "N/A"]
    mean_spearman = np.mean(spearmans) if spearmans else 0.0

    avg_index_time = np.mean([r["index_time_sec"] for r in results_records])
    avg_frames = np.mean([r["indexed_frames"] for r in results_records])
    avg_skipped_static = np.mean([r["skipped_static"] for r in results_records])
    avg_skipped_semantic = np.mean([r["skipped_semantic"] for r in results_records])

    # Save detailed CSV
    csv_path = out_dir / "evaluation_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_records[0].keys()))
        writer.writeheader()
        writer.writerows(results_records)

    # Save summary JSON
    summary_data = {
        "dataset": args.jsonl,
        "evaluated_samples": n,
        "total_eval_time_sec": total_eval_time,
        "metrics": {
            "hit@1": float(mean_hit1),
            "hit@5": float(mean_hit5),
            "hit@10": float(mean_hit10),
            "recall@1_iou0.5": float(mean_r1_iou5),
            "recall@5_iou0.5": float(mean_r5_iou5),
            "recall@1_iou0.7": float(mean_r1_iou7),
            "recall@5_iou0.7": float(mean_r5_iou7),
            "mrr": float(mean_mrr),
            "map": float(mean_map),
            "mean_saliency_spearman": float(mean_spearman),
        },
        "efficiency": {
            "avg_index_time_sec": float(avg_index_time),
            "avg_indexed_frames": float(avg_frames),
            "avg_skipped_static_frames": float(avg_skipped_static),
            "avg_skipped_semantic_frames": float(avg_skipped_semantic),
        },
    }
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Generate Markdown research paper report
    report_md = f"""# Camiq SigLIP Evaluation Report: QVHighlights Benchmark

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Model Architecture:** ViT-B-16-SigLIP (HF Tokenizer context length 64)  
**Evaluated Dataset:** `{args.jsonl}`  
**Evaluated Samples:** {n}  
**Total Wall Time:** {total_eval_time}s  

---

## 1. Summary of Performance Metrics (Research Benchmark)

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Hit@1** | **{mean_hit1 * 100:.2f}%** | Top-1 retrieved frame lands in ground truth window |
| **Hit@5** | **{mean_hit5 * 100:.2f}%** | Top-5 retrieved frames include ground truth window |
| **Hit@10** | **{mean_hit10 * 100:.2f}%** | Top-10 retrieved frames include ground truth window |
| **R@1 (tIoU ≥ 0.5)** | **{mean_r1_iou5 * 100:.2f}%** | Top-1 moment overlap tIoU ≥ 0.5 |
| **R@5 (tIoU ≥ 0.5)** | **{mean_r5_iou5 * 100:.2f}%** | Top-5 moment overlap tIoU ≥ 0.5 |
| **R@1 (tIoU ≥ 0.7)** | **{mean_r1_iou7 * 100:.2f}%** | Strict temporal overlap tIoU ≥ 0.7 |
| **R@5 (tIoU ≥ 0.7)** | **{mean_r5_iou7 * 100:.2f}%** | Strict temporal overlap tIoU ≥ 0.7 |
| **MRR** | **{mean_mrr:.4f}** | Mean Reciprocal Rank across queries |
| **mAP** | **{mean_map:.4f}** | Mean Average Precision across temporal windows |
| **Saliency Spearman $\\rho$** | **{mean_spearman:.4f}** | Rank correlation with human saliency annotations |

---

## 2. Pipeline Efficiency & Frame Filtering Analysis

| Efficiency Metric | Average Value per Video |
| :--- | :--- |
| **Indexing Latency** | {avg_index_time:.2f} s |
| **Retained Frames** | {avg_frames:.1f} frames |
| **Static Frames Pruned** | {avg_skipped_static:.1f} frames |
| **Semantic Duplicates Pruned** | {avg_skipped_semantic:.1f} frames |

---

## 3. Detailed Per-Query Results

| QID | Video ID | Query | GT Windows | Top-1 Time | Score | Hit@1 | Hit@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results_records:
        report_md += f"| {r['qid']} | `{r['vid']}` | {r['query'][:40]}... | {r['gt_windows']} | {r['top1_timestamp']} | {r['top1_score']} | {'✅' if r['hit@1'] else '❌'} | {'✅' if r['hit@5'] else '❌'} | {r['reciprocal_rank']} |\n"

    report_md += f"""
---
*Full per-query raw logs and metrics are exported in `evaluation_details.csv`.*  
*Retrieved frames can be inspected in `{frames_dir.name}/`.*
"""

    report_path = out_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f" Summary Metrics: Hit@1 = {mean_hit1*100:.1f}%, Hit@5 = {mean_hit5*100:.1f}%, MRR = {mean_mrr:.3f}, mAP = {mean_map:.3f}")
    print(f" Generated Reports:")
    print(f"   - Markdown: {report_path.resolve()}")
    print(f"   - CSV:      {csv_path.resolve()}")
    print(f"   - JSON:     {(out_dir / 'summary_metrics.json').resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
