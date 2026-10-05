# SigLIP Dual-Filter Ablation Study

**Dataset:** `highlight_train_release.jsonl` — 30 videos, 30 requested  
**MAD threshold (when active):** `8.0`  
**Semantic threshold (when active):** `0.88`  
**Date:** 2026-10-05 13:20:11

---

## Conditions

| ID | Condition | Stage 1 (MAD) | Stage 2 (Sem-Sim) |
|---|---|---|---|
| A | No filters | disabled | disabled |
| B | Static only | `MAD < 8.0` | disabled |
| C | Semantic only | disabled | `sim > 0.88` |
| D | Both (default) | `MAD < 8.0` | `sim > 0.88` |

---

## Retrieval Quality

| Condition | Hit@1 | Hit@5 | R@1 IoU≥0.5 | R@5 IoU≥0.5 | MRR | mAP | ΔHit@1 | ΔmAP |
|---|---|---|---|---|---|---|---|---|
| [A] No filters         | 0.733 | 0.933 | 0.000 | 0.000 | 0.830 | 0.793 | +0.000 | +0.000 |
| [B] Static only        | 0.733 | 0.933 | 0.033 | 0.067 | 0.834 | 0.780 | +0.000 | -0.013 |
| [C] Semantic only      | 0.767 | 0.933 | 0.033 | 0.067 | 0.852 | 0.761 | +0.033 | -0.032 |
| [D] Both filters       | 0.767 | 0.933 | 0.033 | 0.067 | 0.853 | 0.754 | +0.033 | -0.039 |

*Δ values are relative to the No-filters baseline [A].*

---

## Indexing Efficiency

| Condition | Retention | Frames Kept | SkipStatic | SkipSem | Avg Index Time | Speedup |
|---|---|---|---|---|---|---|
| [A] No filters         | 100.0% | 152 | 0 | 0 | 20.974s | 1.00× |
| [B] Static only        | 68.5% | 105 | 47 | 0 | 14.321s | 1.46× |
| [C] Semantic only      | 26.1% | 40 | 0 | 112 | 21.127s | 0.99× |
| [D] Both filters       | 25.7% | 39 | 47 | 66 | 14.220s | 1.47× |

*Speedup is relative to baseline [A] indexing time.*

---

## Filter Contribution Analysis

| Filter | Frames eliminated (avg) | Retention reduction | Index speedup |
|---|---|---|---|
| Stage 1 — Static (MAD) | 47.4 frames | 31.5% → 0.0% | 1.46× |
| Stage 2 — Semantic (sim) | 112.2 frames | 73.9% → 0.0% | 0.99× |
| Both combined | 112.9 frames | 74.4% → 0.0% | 1.47× |

### Quality trade-off (vs baseline)

| Filter | ΔHit@1 | ΔmAP | Verdict |
|---|---|---|---|
| Stage 1 — Static | +0.000 | -0.013 | ✅ No quality loss |
| Stage 2 — Semantic | +0.033 | -0.032 | ✅ No quality loss |
| Both combined | +0.033 | -0.039 | ✅ No quality loss |

---

*Full per-condition data: `ablation_results.json`*
