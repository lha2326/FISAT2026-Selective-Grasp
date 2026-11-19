# FISAT 2026 — Distribution-Free Rejection Rules for Selective Robotic Grasp Detection

Research archive accompanying the 12-page paper **“Distribution-Free Rejection Rules and Risk-Coverage Analysis for Selective Robotic Grasp Detection”**, EAI FISAT 2026.

**Author (as in PDF):** Hang Anh Le (University of Economics – Technology for Industries, Vietnam).

## Research summary

A *frozen* RGB-D grasp detector can abstain from a low-confidence prediction. For a scalar score where larger means less certain, execute only if `score <= threshold`. The Learn-then-Test calibration procedure seeks an operating threshold under a chosen executed failure-risk budget `alpha` with confidence `1 - delta`. Reported success is the **rectangle metric on static RGB-D images**, not physical robot grasp success.

- Datasets: Cornell and Jacquard V2.
- Architectures: GR-ConvNet, GG-CNN, U-Net.
- Uncertainty scores: peak quality, quality-map entropy, MC dropout, evidential components and top-k disagreement.
- Risk budgets: 0.15, 0.10, 0.05, 0.02, 0.01; reported confidence: 90%.

## Research package (all provided material organized)

| Path | Contents |
|---|---|
| `paper/` | Submitted manuscript (PDF) |
| `src/` | 15 Python files: datasets, models, extraction, calibration, validation |
| `configs/` | Six dataset-model YAML configurations |
| `notebooks/executed/` | Six originally executed Colab notebooks, with the saved outputs |
| `notebooks/templates/` | Five clean/original Colab templates from the source package |
| `data/scores/` | Six score-cache NPZ files (3 detectors × 2 datasets) |
| `checkpoints/` | Best/last GR-ConvNet and evidential-head checkpoints plus training histories for both datasets |
| `results/jacquard/` | Two supplied Jacquard figures |
| `results/cornell_legacy/` | Earlier Cornell tables and figures, **not the final paper tables** |
| `results/summary/` | Newly cross-checked Table 1 metrics and a regenerated diagnostic figure |
| `scripts/` | Cache-to-paper consistency checks and figure regeneration |
| `docs/` | Completeness, provenance and methodological audit |

## Verify the numerical paper results (without datasets/GPU)

```bash
python -m pip install -r requirements.txt
python scripts/verify_paper_metrics.py
python scripts/verify_paper_metrics.py --full  # includes 300 Jacquard re-splits
python scripts/rebuild_figures.py
```

A quick test using an individual cached score file:

```bash
cd src
python selective_grasp.py --scores ../data/scores/jacquard_grconvnet.npz --alpha 0.05 --delta 0.10 --out ../results/local_analysis
```

**Important:** The `src/selective_grasp.py` CLI outputs exploratory Tables 1–3 from a former experiment outline; do not equate them with Tables 1–3 in the final publication. `scripts/verify_paper_metrics.py` is the appropriate consistency check for Table 1 and for the specified Jacquard validation numbers.

## Train models anew

1. Obtain the original Cornell/Jacquard V2 datasets from their original sources (raw image data are not bundled).
2. Edit dataset `data.root` in the corresponding `configs/` YAML.
3. From `src/` run for example:

```bash
python train.py --smoke
python train.py --config ../configs/configs_cornell.yaml --method deterministic
python train.py --config ../configs/configs_cornell.yaml --method evidential
```

The executed Colab notebooks embed module source via `%%writefile` and can be re-run in Colab after adjusting Drive locations. Original paths such as `/content/drive/MyDrive/PhD/...` are historical examples and must be changed for another environment. Full raw-data training runs have **not** been executed as part of this repository packaging.

## Status and limitations

**What is present:** manuscript, code, six score arrays, six executed notebooks, and checkpoints for GR-ConvNet plus evidential models on both datasets. The six cached success rates and best AURCs match the rounded values reported in manuscript Table 1. Jacquard GR-ConvNet 300-split checks were also reproduced from the scores.

**What is not present:** raw Cornell/Jacquard image datasets, checkpoints for the GG-CNN and U-Net experiment runs, explicit dataset file manifests/fold index lists, a pinned environment/lockfile and independent physical-robot validation. Existing score caches suffice for selected metric checks but not a full from-scratch replication of all models. See [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md).

**Publication & sharing:** Confirm the conference/publisher's redistribution terms before putting the manuscript PDF and trained weights in a **public** repository. This archive intentionally does not assign a software license; ownership and reuse terms must be decided by the author.
