# FISAT 2026 — Selective Grasp Detection: Complete Package

*"When Not to Grasp: Distribution-Free Rejection Rules and Risk–Coverage
Analysis for Selective Grasp Detection"*

## What this is

A standalone codebase + Colab notebooks implementing a risk-controlled rejection
rule for grasp detection: given a FROZEN detector and scalar uncertainty scores,
calibrate an abstention threshold so the failure rate among EXECUTED grasps is
<= alpha with probability >= 1 - delta, distribution-free, via Learn-then-Test.

## Notebooks — "train in one place, aggregate in another"

### Training notebooks (heavy, run once each)
- **FISAT2026_cornell_colab.ipynb** — full pipeline on Cornell. Trains
  GR-ConvNet + evidential, caches scores, runs every analysis. Start here.
- **FISAT2026_jacquard_colab.ipynb** — same pipeline on Jacquard V2.
- **FISAT2026_model_agnostic.ipynb** — trains two MORE detectors (GG-CNN,
  U-Net) to show the rule is model-agnostic. Has a `DATASET` switch; run once
  for 'cornell' and once for 'jacquard'.

### Analysis notebooks (light, run anytime — no training)
- **FISAT2026_results_summary.ipynb** — THE aggregation notebook. Loads every
  scores.npz and builds all comparison tables + figures, including the
  per-object simulation figure. Degrades gracefully if some caches are missing.
- **FISAT2026_cross_dataset.ipynb** — focused Cornell-vs-Jacquard regime
  comparison.

## Contributions (each maps to tested code)

1. Distribution-free risk-controlled rejection rule via LTT — `selective_grasp.py`
2. Finding: best-ranking score != best-operating-point score — `object_eval.py`
3. Conjunctive rule, formalized with complementarity analysis — `selective_grasp.py`
4. Learned abstention gate — `gate_head.py`
5. Model-agnostic across 3 detectors x 2 datasets — `models_{grconvnet,ggcnn,unet}.py`
6. Calibration-size law n >= ln(1/delta)/(2 alpha^2) — `selective_grasp.py`
   plus an honest negative result on two attempted new UQ scores (GSU,
   feature-density), which fail on Cornell because its failures are epistemic.

## Modules

| file | role |
|---|---|
| `data_cornell.py`, `data_jacquard.py` | dataset loaders (object-disjoint splits) |
| `models_grconvnet.py`, `models_ggcnn.py`, `models_unet.py` | three detectors |
| `models_evidential.py` | evidential head (frozen-backbone) |
| `metrics_grasp.py` | rectangle-metric grasp success |
| `train.py` | training + `make_model` / `make_dataset_factory` factories |
| `extract_scores.py` | one-pass score cache -> scores.npz |
| `selective_grasp.py` | risk-coverage, LTT, conjunction, calibration theory |
| `gate_head.py` | learned abstention gate |
| `object_eval.py` | per-object eval, simulation, figures |
| `gsu.py`, `test_feature_density.py` | the two attempted new scores (negative result) |
| `make_figures.py` | qualitative grasp figures |
| `configs_*.yaml` | per (dataset x detector) configs |

## How to run

1. Upload the dataset to Drive; point the notebook's DRIVE/DATA paths at it.
2. Run `FISAT2026_cornell_colab.ipynb` end to end -> produces `scores.npz`.
3. (Optional) Run `FISAT2026_jacquard_colab.ipynb` and
   `FISAT2026_model_agnostic.ipynb` for the extra datasets/detectors.
4. Run `FISAT2026_results_summary.ipynb` to assemble every table and figure.

Every module has an offline self-test (`python <module>.py`); the notebooks
embed all modules via %%writefile so they are fully self-contained.
