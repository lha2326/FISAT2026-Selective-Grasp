# Distribution-Free Rejection Rules for Selective Robotic Grasp Detection

**Distribution-Free Rejection Rules and Risk-Coverage Analysis for Selective Robotic Grasp Detection**  
Hang Anh Le · University of Economics–Technology for Industries (UNETI), Vietnam  
EAI FISAT 2026

[Paper](paper/FISAT2026_HangAnhLe.pdf) · [Source code](src/) · [Notebooks](notebooks/) · [Experimental results](results/) · [Reproducibility assessment](docs/REPRODUCIBILITY_AUDIT.md)

## Abstract

This repository accompanies a study of distribution-free risk control for selective robotic grasp detection. A frozen RGB-D grasp detector produces a candidate grasp and an uncertainty score. A threshold-based selection rule permits abstention, and a Learn-then-Test (LTT) calibration procedure aims to control the conditional failure rate among executed grasps at a prescribed risk budget and confidence level. The analysis considers the Cornell and Jacquard V2 datasets, three detector architectures, and alternative uncertainty measures. Evaluation is based on the offline rectangle metric.

## Method

For a candidate grasp with uncertainty score \(s\), execution is determined by

\[
\operatorname{execute}(s;\tau)=\mathbf{1}\{s\leq\tau\}.
\]

The corresponding selective risk and coverage are

\[
R(\tau)=\frac{\mathbb{E}[L\,\mathbf{1}\{s\leq\tau\}]}{\mathbb{E}[\mathbf{1}\{s\leq\tau\}]},
\qquad
C(\tau)=\Pr(s\leq\tau),
\]

where \(L=1\) denotes failure under the rectangle metric. Conditional risk is defined for operating points with positive coverage. The LTT calibration framework evaluates candidate thresholds against a target risk \(\alpha\) at confidence \(1-\delta\), subject to the specified statistical assumptions.

**Experimental settings:** Cornell and Jacquard V2; GR-ConvNet, GG-CNN, and U-Net; confidence \(1-\delta=0.90\); risk budgets \(\alpha\in\{0.15,0.10,0.05,0.02,0.01\}\).

## Experimental results

### Detector performance and uncertainty-score comparison

Reported values correspond to Table 1 of the accompanying manuscript. AURC denotes the area under the risk–coverage curve; lower values indicate better aggregate selective performance.

| Dataset | Detector | Grasp success | Best score by AURC | AURC ↓ | Certified at α = 0.10 | Strictest certified α |
|:--|:--|--:|:--|--:|:--:|--:|
| Cornell | GR-ConvNet | 68.7% | S3: MC-dropout | 0.195 | No | None |
| Cornell | GG-CNN | 63.0% | S3: MC-dropout | 0.199 | No | None |
| Cornell | U-Net | 82.3% | S3: MC-dropout | 0.075 | No | 0.15 |
| Jacquard | GR-ConvNet | 92.8% | S1: peak confidence | 0.016 | Yes | 0.05 |
| Jacquard | GG-CNN | 92.0% | S1: peak confidence | 0.021 | Yes | 0.05 |
| Jacquard | U-Net | 96.0% | S5: top-k disagreement | 0.011 | Yes | 0.05 |

Evaluation sets contain 300 grasps for Cornell and 375 for Jacquard. Score rankings are retrospective observations, not universal score-selection rules.

### Certified coverage versus risk budget

![Certified coverage across detector–dataset configurations](results/summary/coverage_vs_alpha_from_caches.png)

*Figure A. Diagnostic comparison reconstructed from archived score arrays. This plot is not the original figure embedded in the paper. A plotted zero denotes the absence of a certified nonempty operating threshold.*

### Risk–coverage analysis

![Risk–coverage curves for Jacquard](results/jacquard/risk_coverage.png)

*Figure B. Risk–coverage curves for alternative uncertainty scores on Jacquard. The oracle provides a retrospective reference.*

### Selective grasp decisions

![Accepted and rejected grasps on Jacquard](results/jacquard/per_object.png)

*Figure C. Examples from Jacquard. Green borders indicate executed grasps; red borders indicate abstentions.*

### Empirical risk-control verification

The Jacquard/GR-ConvNet setting was evaluated over 300 random calibration/evaluation partitions (manuscript Table 2).

| Risk budget α | Certifiable | Empirical hold rate | Mean executed risk |
|--:|:--:|--:|--:|
| 0.15 | Yes | 1.000 | 0.072 |
| 0.10 | Yes | 1.000 | 0.054 |
| 0.05 | Yes | 0.990 | 0.012 |
| 0.02 | No | — | — |
| 0.01 | No | — | — |

At \(\alpha=0.05\), the reported operating point executed 232 of 375 grasps (coverage **0.619**), including 227 successes and 5 failures. The empirical hold rate (**0.990**) represents the proportion of evaluation partitions satisfying the target risk; it is distinct from grasp coverage. The paper's Hoeffding sample-size expression is a conservative sufficient reference; certification in the reported experiments uses the Hoeffding–Bentkus test.

## Repository structure

| Path | Content |
|:--|:--|
| [`paper/`](paper/) | FISAT manuscript (PDF) |
| [`src/`](src/) | Detector, uncertainty estimation, calibration, and evaluation implementations |
| [`configs/`](configs/) | Model and dataset configurations |
| [`notebooks/executed/`](notebooks/executed/) | Executed Colab notebooks with archived output |
| [`notebooks/templates/`](notebooks/templates/) | Notebook source templates |
| [`data/scores/`](data/scores/) | Six archived score arrays (`.npz`) |
| [`checkpoints/`](checkpoints/) | Eight archived model-weight files (`.pt`) and training histories |
| [`results/jacquard/`](results/jacquard/) | Jacquard risk–coverage and grasp visualizations |
| [`results/summary/`](results/summary/) | Cross-model statistics and diagnostic figures |
| [`results/cornell_legacy/`](results/cornell_legacy/) | Historical Cornell analyses, distinct from final manuscript tables |
| [`scripts/`](scripts/) | Numerical verification and figure-generation scripts |
| [`docs/`](docs/) | Archive provenance and reproducibility assessment |

## Verification

Selected manuscript metrics can be checked from the archived score arrays without retraining the detectors.

```bash
python -m pip install -r requirements.txt
python scripts/verify_paper_metrics.py
python scripts/verify_paper_metrics.py --full
python scripts/rebuild_figures.py
```

The `--full` option includes a 300-partition Jacquard verification. The command-line analysis in `src/selective_grasp.py` follows an earlier exploratory reporting format and is not equivalent to the final manuscript Tables 1–3.

## Reproducibility and scope

The archive contains source code, six score arrays, executed notebooks, and eight supplied checkpoints for deterministic and evidential model runs on Cornell and Jacquard. GG-CNN and U-Net checkpoints for the reported comparison are not available in the archive. Raw dataset images, exact original split manifests, and a fully pinned training-environment lockfile are also absent.

The reported outcomes concern grasp detection under the offline rectangle metric. Exchangeability is assumed for the statistical risk-control formulation. Sequential bin-picking distribution shift, contact dynamics, collision constraints, and physical robot execution were not evaluated. Additional validation is required before interpreting these results as physical grasp-success guarantees.

For file-level provenance and coverage of the numerical checks, see [the reproducibility assessment](docs/REPRODUCIBILITY_AUDIT.md).

## License

No software license is specified for this repository.
