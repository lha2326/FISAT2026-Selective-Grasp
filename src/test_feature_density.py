"""
test_feature_density.py — GO / NO-GO test for a new UQ score, BEFORE building it
================================================================================
The lesson from GSU: do not write a whole module and wire it through two
notebooks only to discover the dataset doesn't contain the failure mode the
score targets. This script tests the hypothesis on your REAL cached data first.

Hypothesis
----------
Cornell's strongest existing score is S3 (MC-dropout), which measures EPISTEMIC
uncertainty -- the model being unsure because it hasn't seen enough data. That
points to failures being driven by out-of-distribution grasps, not by peak shape
(GSU) or multi-peak ambiguity (S5). A FEATURE-DENSITY score reads the same
epistemic signal as S3 but at one forward pass: a grasp whose features are far
from the bulk of other grasps is one the detector has little support for.

What this script does
---------------------
Loads scores.npz (which already contains `feat`), computes the feature-density
score, and reports its correlation with failure and its AURC against the
existing scores. It builds NOTHING into the pipeline. Read the verdict at the
bottom:

    GO      -- density correlates with failure and its AURC beats S5 (the
               weakest real score); worth building into a full contribution.
    NO-GO   -- density is weak too; Cornell failures aren't OOD-shaped either.
               Report the negative result and lean on the main contribution.

Usage
-----
    python test_feature_density.py --scores /path/to/scores.npz

If you don't pass --scores it looks for ./scores.npz.
"""

from __future__ import annotations

import argparse
import numpy as np

EPS = 1e-9


def feature_density_score(feat, k=10):
    """Mean standardized distance to the k nearest neighbours in feature space.

    Higher = more isolated = less in-distribution = less trustworthy (matches
    the pipeline convention: accept iff score <= tau).
    """
    feat = np.asarray(feat, dtype=float)
    if feat.ndim != 2 or feat.shape[0] < k + 1:
        raise ValueError(f"feat must be (N, D) with N > k; got {feat.shape}")
    mu, sd = feat.mean(0), feat.std(0) + EPS
    z = (feat - mu) / sd
    n = len(z)
    out = np.zeros(n)
    # chunked to stay memory-friendly for large N
    for i in range(n):
        d = np.sqrt(((z - z[i]) ** 2).sum(1))
        d[i] = np.inf
        out[i] = np.sort(d)[:k].mean()
    return out


def mahalanobis_density(feat):
    """Alternative: distance to the global mean under the feature covariance.

    Cheaper than kNN (no pairwise), and a classic OOD proxy. Reported alongside
    kNN so the test doesn't hinge on one estimator.
    """
    feat = np.asarray(feat, dtype=float)
    mu = feat.mean(0)
    cov = np.cov(feat.T) + EPS * np.eye(feat.shape[1])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    d = feat - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def aurc(score, loss):
    """Area under the risk-coverage curve (lower = better)."""
    order = np.argsort(score)
    l = np.asarray(loss)[order]
    cov_risk = np.cumsum(l) / (np.arange(len(l)) + 1)
    return float(cov_risk.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="scores.npz")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.scores)
    if "feat" not in d:
        raise SystemExit(
            "scores.npz has no 'feat' array. Re-run extract_scores with the "
            "version that caches features (it stores rec['feat']).")
    feat = d["feat"]
    success = d["success"].astype(int)
    loss = 1 - success
    print(f"loaded {len(success)} grasps, feat shape {feat.shape}, "
          f"success rate {success.mean()*100:.1f}%\n")

    knn = feature_density_score(feat, k=args.k)
    maha = mahalanobis_density(feat)

    from numpy import corrcoef
    c_knn = corrcoef(knn, loss)[0, 1]
    c_maha = corrcoef(maha, loss)[0, 1]

    # existing scores for comparison
    print("correlation with failure (higher positive = better):")
    print(f"  feature-density kNN    {c_knn:+.3f}")
    print(f"  feature-density Maha   {c_maha:+.3f}")
    for key, label in [("s_conf", "S1 peak conf"), ("s_mc", "S3 MC-dropout"),
                       ("s_topk", "S5 top-k")]:
        if key in d:
            print(f"  {label:20s} {corrcoef(d[key], loss)[0,1]:+.3f}")
    print()

    print("AURC (lower = better):")
    rows = [("feature-density kNN", aurc(knn, loss)),
            ("feature-density Maha", aurc(maha, loss))]
    for key, label in [("s_conf", "S1 peak conf"), ("s_mc", "S3 MC-dropout"),
                       ("s_topk", "S5 top-k"), ("s_gsu", "GSU")]:
        if key in d:
            rows.append((label, aurc(d[key], loss)))
    rows.sort(key=lambda r: r[1])
    for label, a in rows:
        print(f"  {label:20s} {a:.4f}")
    print()

    # verdict
    best_density_aurc = min(aurc(knn, loss), aurc(maha, loss))
    s5_aurc = aurc(d["s_topk"], loss) if "s_topk" in d else np.inf
    best_c = max(c_knn, c_maha)
    print("=" * 60)
    if best_c > 0.15 and best_density_aurc < s5_aurc:
        print("VERDICT: GO")
        print("  Feature-density tracks failure and beats S5. This epistemic")
        print("  score is worth building into a full contribution: wire it into")
        print("  extract_scores, add it to SCORE_KEYS, and test it as a")
        print("  conjunctive partner with S3.")
    else:
        print("VERDICT: NO-GO")
        print("  Feature-density is weak on this data too. Cornell failures are")
        print("  not OOD-shaped in this feature space. Report this honestly as a")
        print("  second negative result and lean on the main contribution")
        print("  (risk-controlled rejection), which does not need a new score.")
        print("  Consider running this same test on the Jacquard cache before")
        print("  abandoning the idea -- a larger, more varied dataset may show")
        print("  the OOD signal Cornell lacks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
