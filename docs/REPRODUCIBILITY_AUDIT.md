# Reproducibility / completeness audit — 16 November 2026

## What was checked

The 12-page supplied FISAT PDF was compared with the four uploaded ZIP archives. Counts are by distinct configurations, not duplicated notebook source. All Python `.py` source modules compiled; offline module self-tests (`data_jacquard`, `metrics_grasp`, `models_grconvnet`, `models_ggcnn`, `models_unet`, `models_evidential`, `train --smoke`, `object_eval`, `gate_head`, `gsu`) returned PASS. This is **not** an end-to-end retraining check.

## Check against manuscript Table 1

| Dataset | Model | N | success in supplied NPZ | Best AURC score | Recalculated AURC | PDF AURC (rounded) | Result |
|---|---|---:|---:|---|---:|---:|---|
| Cornell | GR-ConvNet | 300 | 68.67% | MC dropout | 0.1954 | 0.195 | pass |
| Cornell | GG-CNN | 300 | 63.00% | MC dropout | 0.1990 | 0.199 | pass |
| Cornell | U-Net | 300 | 82.33% | MC dropout | 0.0746 | 0.075 | pass |
| Jacquard | GR-ConvNet | 375 | 92.80% | peak quality | 0.0163 | 0.016 | pass |
| Jacquard | GG-CNN | 375 | 92.00% | peak quality | 0.0205 | 0.021 | pass |
| Jacquard | U-Net | 375 | 96.00% | top-k | 0.0106 | 0.011 | pass |

With Jacquard GR-ConvNet `s_conf`, a full-cache LTT run at `alpha=0.05`, `delta=0.10` certified 232/375 grasps (coverage 0.6187); this agrees with the paper's 0.619 and the 227 successes/5 failures in Table 3. The 300 random 50:50 re-splits in the supplied code (`seed=0`) yielded hold rates 1.000 at alpha 0.10 and 0.990 at alpha 0.05. The mean evaluated risks were approx. 0.0541 and 0.0116 respectively (matching the publication's rounded Table 2). These are *empirical checks from the cached dataset*, not a new independent validation study.

## Missing or unresolved items

| Priority | Item | Finding | What is needed for complete reproducibility |
|---|---|---|---|
| High | GG-CNN/U-Net checkpoints | No `.pt` checkpoint included for these four trained model configurations | Export best/last `.pt` plus training histories for each dataset |
| High | Raw datasets / access instructions | Cornell and Jacquard V2 images and annotations not in uploads | Link to official dataset sources, instructions to download and prepare, optionally checksums |
| High | Fixed split manifests | Cache contains indices/object_id but no explicit original train/validation/calibration/test filename manifests | Export exact per-split sample IDs and protocol; clarify object-disjoint versus image-wise resampling used by later analyses |
| Medium | Environment | No pinned versions or lockfile in uploaded artifacts | Record training environment (Python, torch, torchvision, CUDA, numpy, scipy etc.) and seed settings |
| Medium | Paper numerical provenance | Manuscript Table 2 uses re-splitting of already cached test scores; full data extraction pathway was not re-run | Preserve generated intermediate logs and a fixed-command pipeline that exactly creates each publication table/figure |
| Medium | Theoretical guarantee and selection protocol | Score ranking / choice and threshold selection operate on a limited cache; finite-sample risk guarantees rely on their stated assumptions and correct testing protocol | Independent methodology review, held-out development set for choosing score, assess candidate-grid construction and exchangeability assumptions |
| Low | Figure provenance | Existing Jacquard risk–coverage and per-object figures are supplied; manuscript PDF has its own figures | Retain plot inputs and exact scripts/settings for the final three paper figures |
| Low | Publishing license | None specified | Confirm publisher rights and decide code/data/checkpoint license before making public |

## Historical material diverging from final manuscript

The original `PACKAGE_README.md` and `results/cornell_legacy` describe conjunctive rejection and learned gates among proposed contributions. The supplied 12-page PDF does **not** list these as central contributions. They have been retained for archival integrity but are **not labeled as paper-confirmed experiments**. The generated source CLI's `tables.md` also has a different structure and must not be confused with Tables 1–4 in the paper.

## Cautions about interpreting the results

- The publication evaluates static RGB-D rectangle metric success. It does **not** demonstrate actual hardware grasp success or robustness under distribution shift.
- A saved NPZ can validate computations from its scores, but is not independent evidence that the original detector training/data-preparation process can be reproduced from scratch.
- The per-split 'hold rate' in the original notebook is not the same quantity as grasp execution coverage; no-certification splits are skipped by `verify_risk_guarantee`.
- The existing cached set is also re-used for analysis (and for selecting favorable scores); before advertising a formal risk guarantee beyond the paper's stated conditions, a statistical review is recommended.
- The source package supplied a Hoeffding sample-size reference. It is **not** a universal minimum size; the actual calibration uses the different Hoeffding–Bentkus procedure and observed loss values.
