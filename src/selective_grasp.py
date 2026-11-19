"""
selective_grasp.py
==================
Rejection rules and risk-coverage analysis for grasp prediction.

Input contract
--------------
This module deliberately knows nothing about GR-ConvNet. It consumes a cached
.npz produced by `extract_scores.py` containing, for N test samples:

    success : (N,) int/bool   1 if the predicted grasp satisfies the rectangle
                              metric (IoU > 0.25 and angle error < 30 deg)
    s_conf  : (N,) float      S1  1 - peak quality
    s_ent   : (N,) float      S2  spatial entropy of the quality map
    s_mc    : (N,) float      S3  MC-dropout pose dispersion      (optional)
    s_ev_ep : (N,) float      S4  evidential epistemic dispersion (optional)
    s_ev_al : (N,) float      S4  evidential aleatoric dispersion (optional)
    s_topk  : (N,) float      S5  top-k grasp disagreement

Convention: for every score, LARGER = LESS TRUSTWORTHY. Accept iff s <= tau.

Usage
-----
    python selective_grasp.py --scores scores.npz --alpha 0.05 --delta 0.10 \
                              --out results/

Outputs Table 1 / Table 2 / Table 3 of the paper as CSV + markdown, and the
risk-coverage figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import binom

EPS = 1e-12

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# 1. Risk-coverage machinery
# --------------------------------------------------------------------------- #

def risk_coverage_curve(scores: np.ndarray, loss: np.ndarray):
    """Full risk-coverage curve, evaluated at every achievable coverage.

    Parameters
    ----------
    scores : (N,) larger = less trustworthy
    loss   : (N,) in [0,1]; here the 0/1 failure indicator

    Returns
    -------
    coverage : (N,) increasing from 1/N to 1
    risk     : (N,) selective risk at each coverage
    """
    order = np.argsort(scores, kind="mergesort")     # ascending uncertainty
    l_sorted = loss[order]
    n = len(loss)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(l_sorted) / np.arange(1, n + 1)
    return coverage, risk


def aurc(scores: np.ndarray, loss: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage_curve(scores, loss)
    return float(np.mean(risk))


def oracle_aurc(loss: np.ndarray) -> float:
    """AURC of the hindsight-optimal gate (sort by true loss)."""
    return aurc(loss.astype(float), loss)


def excess_aurc(scores: np.ndarray, loss: np.ndarray) -> float:
    return aurc(scores, loss) - oracle_aurc(loss)


def risk_at_coverage(scores: np.ndarray, loss: np.ndarray, cov: float) -> float:
    """Selective risk when the gate accepts the `cov` fraction of lowest scores."""
    coverage, risk = risk_coverage_curve(scores, loss)
    idx = int(np.clip(np.searchsorted(coverage, cov), 0, len(risk) - 1))
    return float(risk[idx])


def coverage_at_risk(scores: np.ndarray, loss: np.ndarray, alpha: float,
                     n_min: int = 30) -> float:
    """Largest coverage whose empirical selective risk stays <= alpha.

    Returns 0.0 if no threshold with at least `n_min` accepted samples works.
    """
    coverage, risk = risk_coverage_curve(scores, loss)
    n = len(loss)
    valid = (risk <= alpha) & (np.arange(1, n + 1) >= n_min)
    return float(coverage[valid].max()) if valid.any() else 0.0


def ause(scores: np.ndarray, loss: np.ndarray) -> float:
    """Area under the sparsification error curve.

    IMPORTANT: with a *binary* loss the sparsification curve is exactly the
    reversed risk-coverage curve, and since sp_score >= sp_oracle pointwise,
    this quantity is algebraically identical to E-AURC. The column is therefore
    redundant here. To make the paper's ranking-vs-operating-point contrast
    real, compute AUSE on a *continuous* error (IoU deficit, angle error, or
    the regression residual used in the ICARCV benchmark) by passing that array
    as `loss` instead of the 0/1 indicator. Then AUSE and AURC genuinely
    measure different things and may disagree.
    """
    n = len(loss)
    order_s = np.argsort(-scores, kind="mergesort")   # remove most uncertain first
    order_o = np.argsort(-loss.astype(float), kind="mergesort")

    def sparsification(order):
        l = loss[order]
        # mean loss of the remaining set after removing the first k
        rem = np.cumsum(l[::-1])[::-1]
        counts = np.arange(n, 0, -1)
        return rem / counts

    sp_s = sparsification(order_s)
    sp_o = sparsification(order_o)
    return float(np.mean(np.abs(sp_s - sp_o)))


# --------------------------------------------------------------------------- #
# 2. Distribution-free risk control (Learn-then-Test)
# --------------------------------------------------------------------------- #

def _h1(a: float, b: float) -> float:
    """KL between two Bernoullis, used by the Hoeffding bound."""
    a = min(max(a, EPS), 1 - EPS)
    b = min(max(b, EPS), 1 - EPS)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_p_value(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R > alpha, with bounded loss in [0,1].

    Small p means we can safely reject H0, i.e. the true risk is <= alpha.
    """
    if n <= 0:
        return 1.0
    if risk_hat >= alpha:
        return 1.0
    p_hoeff = float(np.exp(-n * _h1(risk_hat, alpha)))
    p_bent = float(np.e * binom.cdf(np.ceil(n * risk_hat), n, alpha))
    return min(1.0, min(p_hoeff, p_bent))


@dataclass
class CalibrationResult:
    tau: float | tuple
    p_value: float
    cal_risk: float
    cal_coverage: float
    n_accepted: int
    valid: bool          # False when no threshold could be certified


def ltt_calibrate_single(scores: np.ndarray, loss: np.ndarray,
                         alpha: float, delta: float,
                         n_grid: int = 200, n_min: int = 50) -> CalibrationResult:
    """Fixed-sequence Learn-then-Test for a single-threshold rejection rule.

    Walks candidate thresholds from most conservative (lowest coverage) to most
    liberal, rejecting H_j: R(tau_j) > alpha while p_j <= delta, and stops at
    the first failure. Guarantees P(R(tau_hat) <= alpha) >= 1 - delta.
    """
    qs = np.linspace(0.0, 1.0, n_grid + 1)[1:]           # increasing coverage
    grid = np.quantile(scores, qs)
    grid = np.unique(grid)                                # keeps ascending order

    best = CalibrationResult(np.nan, 1.0, np.nan, 0.0, 0, False)
    for tau in grid:
        acc = scores <= tau
        n_acc = int(acc.sum())
        if n_acc < n_min:
            continue                                      # too few to certify
        r_hat = float(loss[acc].mean())
        p = hb_p_value(r_hat, n_acc, alpha)
        if p <= delta:
            best = CalibrationResult(float(tau), p, r_hat,
                                     n_acc / len(scores), n_acc, True)
        else:
            break                                         # fixed-sequence stop
    return best


def ltt_calibrate_conjunctive(score_a: np.ndarray, score_b: np.ndarray,
                              loss: np.ndarray, alpha: float, delta: float,
                              n_grid: int = 20, n_min: int = 50,
                              path_frac: float = 0.5,
                              rng: np.random.Generator | None = None
                              ) -> CalibrationResult:
    """LTT for the conjunctive rule  accept iff (s_a <= t_a) and (s_b <= t_b).

    The threshold space is 2-D so no total order exists. We split the
    calibration set: the first half only *orders* the candidate pairs by their
    coverage (defining a fixed sequence), the second half runs the hypothesis
    tests. Ordering on independent data keeps the sequence pre-specified with
    respect to the testing data, which is what fixed-sequence testing requires.
    """
    rng = rng or np.random.default_rng(0)
    n = len(loss)
    perm = rng.permutation(n)
    n_path = int(path_frac * n)
    idx_path, idx_test = perm[:n_path], perm[n_path:]

    qs = np.linspace(0.0, 1.0, n_grid + 1)[1:]
    grid_a = np.unique(np.quantile(score_a, qs))
    grid_b = np.unique(np.quantile(score_b, qs))
    pairs = list(itertools.product(grid_a, grid_b))

    # order the sequence by coverage measured on the path half only
    cov_path = [
        float(((score_a[idx_path] <= ta) & (score_b[idx_path] <= tb)).mean())
        for ta, tb in pairs
    ]
    order = np.argsort(cov_path)                     # conservative -> liberal

    sa, sb, ls = score_a[idx_test], score_b[idx_test], loss[idx_test]
    best = CalibrationResult((np.nan, np.nan), 1.0, np.nan, 0.0, 0, False)
    for j in order:
        ta, tb = pairs[j]
        acc = (sa <= ta) & (sb <= tb)
        n_acc = int(acc.sum())
        if n_acc < n_min:
            continue
        r_hat = float(ls[acc].mean())
        p = hb_p_value(r_hat, n_acc, alpha)
        if p <= delta:
            best = CalibrationResult((float(ta), float(tb)), p, r_hat,
                                     n_acc / len(ls), n_acc, True)
        else:
            break
    return best


def naive_threshold(scores: np.ndarray, loss: np.ndarray, alpha: float,
                    n_min: int = 50) -> float:
    """The uncalibrated baseline: largest tau with empirical cal risk <= alpha.

    Included precisely to show it over-covers and violates alpha on fresh data.
    """
    order = np.argsort(scores, kind="mergesort")
    s_sorted, l_sorted = scores[order], loss[order]
    run = np.cumsum(l_sorted) / np.arange(1, len(loss) + 1)
    ok = (run <= alpha) & (np.arange(1, len(loss) + 1) >= n_min)
    return float(s_sorted[ok][-1]) if ok.any() else float(s_sorted[0])


# --------------------------------------------------------------------------- #
# 3. Logistic fusion (rule R3)
# --------------------------------------------------------------------------- #

def fit_logistic_fusion(S_cal: np.ndarray, loss_cal: np.ndarray,
                        n_iter: int = 500, lr: float = 0.5, l2: float = 1e-3):
    """Plain gradient-descent logistic regression predicting failure.

    Kept dependency-free and tiny on purpose; swap for sklearn if preferred.
    Returns (w, b, mu, sd) where mu/sd standardise the score matrix.
    """
    mu, sd = S_cal.mean(0), S_cal.std(0) + EPS
    X = (S_cal - mu) / sd
    y = loss_cal.astype(float)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def apply_logistic_fusion(S: np.ndarray, params) -> np.ndarray:
    w, b, mu, sd = params
    z = ((S - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-z))          # predicted failure prob = score


# --------------------------------------------------------------------------- #
# 4. Experiment drivers -> the three tables of the paper
# --------------------------------------------------------------------------- #

SCORE_KEYS = [
    ("s_conf",  "S1 negative peak quality"),
    ("s_ent",   "S2 quality-map entropy"),
    ("s_mc",    "S3 MC-dropout dispersion"),
    ("s_ev_ep", "S4 evidential (epistemic)"),
    ("s_ev_al", "S4 evidential (aleatoric)"),
    ("s_topk",  "S5 top-k disagreement"),
]


def load_scores(path: str):
    d = np.load(path)
    loss = 1.0 - d["success"].astype(float)
    scores = {k: d[k].astype(float) for k, _ in SCORE_KEYS if k in d}
    if not scores:
        raise ValueError("no recognised score arrays found in %s" % path)
    return scores, loss


def table1(scores: dict, loss: np.ndarray, alphas=(0.10, 0.05)):
    """Which score gives the best operating curve."""
    rows = []
    for key, name in SCORE_KEYS:
        if key not in scores:
            continue
        s = scores[key]
        row = {
            "score": name,
            "AURC": aurc(s, loss),
            "E-AURC": excess_aurc(s, loss),
            "AUSE": ause(s, loss),
            "R@90": risk_at_coverage(s, loss, 0.90),
            "R@70": risk_at_coverage(s, loss, 0.70),
        }
        for a in alphas:
            row[f"C*(a={a})"] = coverage_at_risk(s, loss, a)
        rows.append(row)
    rows.append({"score": "Oracle", "AURC": oracle_aurc(loss), "E-AURC": 0.0,
                 "AUSE": 0.0, "R@90": risk_at_coverage(loss, loss, 0.90),
                 "R@70": risk_at_coverage(loss, loss, 0.70),
                 **{f"C*(a={a})": coverage_at_risk(loss, loss, a) for a in alphas}})
    return rows


def table2(scores: dict, loss: np.ndarray, alpha: float = 0.05,
           rng: np.random.Generator | None = None):
    """Do combinations help: R1 vs conjunctive R2 vs fused R3."""
    rng = rng or np.random.default_rng(0)
    rows = []

    # R1: best single score by AURC
    singles = {k: aurc(v, loss) for k, v in scores.items()}
    best_key = min(singles, key=singles.get)
    rows.append({"rule": "R1 best single", "scores": best_key,
                 "AURC": singles[best_key],
                 f"C*(a={alpha})": coverage_at_risk(scores[best_key], loss, alpha)})

    # R2: conjunctions of s_conf with each dispersion-style score
    base = "s_conf"
    if base in scores:
        for other in ("s_mc", "s_topk", "s_ev_ep", "s_ent"):
            if other not in scores:
                continue
            cov, risk_c = _conjunctive_curve(scores[base], scores[other], loss)
            rows.append({
                "rule": "R2 conjunctive",
                "scores": f"{base} AND {other}",
                "AURC": float(_trapz(risk_c, cov) / max(cov[-1] - cov[0], EPS)),
                f"C*(a={alpha})": float(cov[risk_c <= alpha].max())
                if (risk_c <= alpha).any() else 0.0,
            })

    # R3: logistic fusion, fitted on half and evaluated on the other half
    keys = [k for k, _ in SCORE_KEYS if k in scores]
    S = np.stack([scores[k] for k in keys], axis=1)
    n = len(loss)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    params = fit_logistic_fusion(S[a], loss[a])
    s_fused = apply_logistic_fusion(S[b], params)
    rows.append({"rule": "R3 logistic fusion", "scores": "+".join(keys),
                 "AURC": aurc(s_fused, loss[b]),
                 f"C*(a={alpha})": coverage_at_risk(s_fused, loss[b], alpha)})
    return rows


def _conjunctive_curve(sa: np.ndarray, sb: np.ndarray, loss: np.ndarray,
                       n_grid: int = 25):
    """Pareto risk-coverage frontier of the conjunctive rule over a 2-D grid."""
    qs = np.linspace(0.02, 1.0, n_grid)
    ga, gb = np.quantile(sa, qs), np.quantile(sb, qs)
    pts = []
    for ta, tb in itertools.product(ga, gb):
        acc = (sa <= ta) & (sb <= tb)
        k = int(acc.sum())
        if k < 30:
            continue
        pts.append((k / len(loss), float(loss[acc].mean())))
    pts.sort()
    cov = np.array([p[0] for p in pts])
    rk = np.array([p[1] for p in pts])
    # lower envelope: best (lowest) risk achievable at each coverage or above
    rk = np.minimum.accumulate(rk[::-1])[::-1]
    return cov, rk


def table3(scores: dict, loss: np.ndarray, alpha: float, delta: float,
           n_splits: int = 100, seed: int = 0):
    """Does the calibrated threshold hold, and what does validity cost."""
    rng = np.random.default_rng(seed)
    n = len(loss)
    key = min(scores, key=lambda k: aurc(scores[k], loss))
    s = scores[key]

    rec = {"naive": {"risk": [], "cov": [], "viol": []},
           "ltt": {"risk": [], "cov": [], "viol": []}}

    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2:]

        t_naive = naive_threshold(s[cal], loss[cal], alpha)
        res_ltt = ltt_calibrate_single(s[cal], loss[cal], alpha, delta)

        for name, tau in (("naive", t_naive),
                          ("ltt", res_ltt.tau if res_ltt.valid else -np.inf)):
            acc = s[ev] <= tau
            if acc.sum() == 0:
                rec[name]["risk"].append(0.0)
                rec[name]["cov"].append(0.0)
                rec[name]["viol"].append(0.0)
                continue
            r = float(loss[ev][acc].mean())
            rec[name]["risk"].append(r)
            rec[name]["cov"].append(float(acc.mean()))
            rec[name]["viol"].append(float(r > alpha))

    rows = []
    for name in ("naive", "ltt"):
        rows.append({
            "rule": f"{name} (score={key})",
            "alpha": alpha,
            "delta": delta if name == "ltt" else None,
            "mean_eval_risk": float(np.mean(rec[name]["risk"])),
            "violation_rate": float(np.mean(rec[name]["viol"])),
            "mean_coverage": float(np.mean(rec[name]["cov"])),
            "coverage_sd": float(np.std(rec[name]["cov"])),
        })
    rows.append({
        "rule": "coverage lost to validity",
        "alpha": alpha, "delta": delta,
        "mean_eval_risk": None, "violation_rate": None,
        "mean_coverage": rows[0]["mean_coverage"] - rows[1]["mean_coverage"],
        "coverage_sd": None,
    })
    return rows


# --------------------------------------------------------------------------- #
# 5. Reporting
# --------------------------------------------------------------------------- #

def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(empty)\n"
    cols = list(rows[0].keys())
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def plot_curves(scores: dict, loss: np.ndarray, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable, skipping figure")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, name in SCORE_KEYS:
        if key not in scores:
            continue
        cov, rk = risk_coverage_curve(scores[key], loss)
        m = cov >= 0.05
        ax.plot(cov[m], rk[m], lw=1.4, label=name)
    cov, rk = risk_coverage_curve(loss, loss)
    m = cov >= 0.05
    ax.plot(cov[m], rk[m], "k--", lw=1.0, label="Oracle")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (failure rate among executed)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"[ok] figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="path to scores .npz")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scores, loss = load_scores(args.scores)
    print(f"[info] N={len(loss)}  base failure rate={loss.mean():.4f}")

    t1 = table1(scores, loss)
    t2 = table2(scores, loss, alpha=args.alpha)
    t3 = table3(scores, loss, args.alpha, args.delta, n_splits=args.splits)
    t3 += table3(scores, loss, 0.10, args.delta, n_splits=args.splits)

    md = ("## Table 1 - score comparison\n\n" + to_markdown(t1) +
          "\n## Table 2 - rule comparison\n\n" + to_markdown(t2) +
          "\n## Table 3 - calibration validity\n\n" + to_markdown(t3))
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "tables.json"), "w") as f:
        json.dump({"table1": t1, "table2": t2, "table3": t3}, f, indent=2)
    plot_curves(scores, loss, os.path.join(args.out, "risk_coverage.png"))
    print(md)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# 6. Paired significance testing for the score/rule comparisons
# --------------------------------------------------------------------------- #
#
# Tables 1 and 2 are comparisons, and the paper's headline claim IS a comparison
# ("the score that wins on AUSE loses on C*"). A bare point estimate cannot
# support that claim. Every score is evaluated on the SAME samples, so the
# comparisons must be PAIRED: resample sample indices once per replicate and
# recompute both metrics on that same resample. Pairing removes the shared
# dataset noise and gives far tighter intervals than treating the two numbers as
# independent estimates.

_METRICS = {
    "AURC": lambda s, l, **kw: aurc(s, l),
    "E-AURC": lambda s, l, **kw: excess_aurc(s, l),
    "C*": lambda s, l, alpha=0.05, n_min=30: coverage_at_risk(s, l, alpha, n_min),
    "R@cov": lambda s, l, cov=0.8, **kw: risk_at_coverage(s, l, cov),
}


def paired_bootstrap_ci(score_a: np.ndarray, score_b: np.ndarray,
                        loss: np.ndarray, metric: str = "AURC",
                        n_boot: int = 2000, conf: float = 0.95,
                        seed: int = 0, **metric_kw):
    """Paired bootstrap CI for metric(a) - metric(b).

    Returns dict with the observed difference, the percentile interval, and
    whether the interval excludes zero. If it does not exclude zero, the two
    scores are not distinguishable at this sample size and the paper must say so
    rather than declaring a winner.
    """
    rng = np.random.default_rng(seed)
    fn = _METRICS[metric]
    n = len(loss)
    obs = fn(score_a, loss, **metric_kw) - fn(score_b, loss, **metric_kw)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                 # same idx for both -> paired
        l = loss[idx]
        if l.sum() == 0 or l.sum() == len(l):       # degenerate resample
            diffs[b] = np.nan
            continue
        diffs[b] = fn(score_a[idx], l, **metric_kw) - fn(score_b[idx], l, **metric_kw)

    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return dict(metric=metric, observed_diff=float(obs),
                ci_lo=float(lo), ci_hi=float(hi),
                significant=bool(lo > 0 or hi < 0), n_boot=len(diffs))


def paired_permutation_test(score_a: np.ndarray, score_b: np.ndarray,
                            loss: np.ndarray, metric: str = "AURC",
                            n_perm: int = 5000, seed: int = 0, **metric_kw):
    """Two-sided paired permutation test for H0: the two scores are equivalent.

    Under H0 the label 'which score' is exchangeable within each sample, so we
    swap the two scores independently per sample and rebuild the null
    distribution of the metric difference. Stricter than the bootstrap and the
    right test for the headline pair, since H0 here is genuine equivalence
    rather than a statement about the size of an effect.
    """
    rng = np.random.default_rng(seed)
    fn = _METRICS[metric]
    n = len(loss)
    obs = fn(score_a, loss, **metric_kw) - fn(score_b, loss, **metric_kw)

    null = np.empty(n_perm)
    for p in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap, score_b, score_a)
        b = np.where(swap, score_a, score_b)
        null[p] = fn(a, loss, **metric_kw) - fn(b, loss, **metric_kw)

    # +1 correction keeps the p-value valid (never reports exactly 0)
    p_val = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return dict(metric=metric, observed_diff=float(obs), p_value=float(p_val),
                significant=bool(p_val < 0.05), n_perm=n_perm)


def random_aurc(loss: np.ndarray) -> float:
    """AURC of a gate that abstains at random: just the base error rate.

    A random score orders nothing, so its selective risk equals the overall
    failure rate at every coverage. This is the floor every real score must beat.
    """
    return float(loss.mean())


def normalized_aurc(scores: np.ndarray, loss: np.ndarray) -> float:
    """AURC rescaled to [0,1]: 0 = oracle, 1 = random. Lower is better.

    This is the number to report when someone asks "how good is this score?"
    Raw AURC depends on the base error rate, so it is not comparable across
    datasets or splits; the normalized version is. A score at 0.5 has closed
    half the gap between random guessing and the hindsight-optimal gate.
    """
    a = aurc(scores, loss)
    o = oracle_aurc(loss)
    r = random_aurc(loss)
    if r - o < 1e-9:
        return 0.0
    return float((a - o) / (r - o))


def baseline_comparison(scores: dict, loss: np.ndarray):
    """One table answering 'is this good?' against oracle and random anchors.

    Reports raw AURC, normalized AURC (0=oracle, 1=random), and the fraction of
    the oracle-to-random gap each score closes. The 'gap closed' column is the
    single most interpretable quality number: 60% means the score recovers 60%
    of the achievable improvement over guessing.
    """
    o = oracle_aurc(loss)
    r = random_aurc(loss)
    rows = []
    for key in list(scores) + ["__oracle__", "__random__"]:
        if key == "__oracle__":
            a, name = o, "ORACLE (upper bound)"
        elif key == "__random__":
            a, name = r, "RANDOM (lower bound)"
        else:
            a, name = aurc(scores[key], loss), key
        norm = 0.0 if (r - o) < 1e-9 else (a - o) / (r - o)
        rows.append({
            "score": name,
            "AURC": a,
            "norm_AURC": norm,               # 0 oracle, 1 random
            "gap_closed_%": 100.0 * (1.0 - norm),
        })
    rows.sort(key=lambda x: x["AURC"])
    return rows


def min_calibration_size(alpha, delta, target_coverage=0.5, n_grid=None):
    """Smallest calibration set that can certify risk <= alpha at confidence
    1 - delta via the Hoeffding bound, as a function of the budget.

    Learn-then-Test with a Hoeffding-Bentkus p-value rejects a threshold when
    the empirical risk r_hat on n calibration points satisfies, roughly,

        r_hat + sqrt( ln(1/delta) / (2 n) )  <=  alpha.

    In the best case r_hat -> 0 (a threshold that makes no mistakes on the
    calibration set), so certification needs the Hoeffding slack alone to fit
    under alpha:

        sqrt( ln(1/delta) / (2 n) )  <=  alpha
        =>  n  >=  ln(1/delta) / (2 alpha^2).

    This closed form is the paper's calibration-size result: it says the
    required calibration set grows as 1/alpha^2, which is exactly why a strict
    budget (alpha = 0.02) needs ~25x more data than a loose one (alpha = 0.10),
    and why Cornell (tens of calibration samples) cannot certify what Jacquard
    (thousands) can. Returns the minimum n (ceil).
    """
    import math
    if alpha <= 0:
        return float("inf")
    n_min = math.log(1.0 / delta) / (2.0 * alpha ** 2)
    return int(math.ceil(n_min))


def calibration_size_table(alphas, deltas):
    """The 1/alpha^2 growth as a table, for the paper's theory section."""
    rows = []
    for a in alphas:
        row = {"alpha": a}
        for d in deltas:
            row[f"n_min(delta={d})"] = min_calibration_size(a, d)
        rows.append(row)
    return rows


def conjunction_complementarity(score_a, score_b, loss):
    """Quantify WHY a conjunction of two scores helps: it helps exactly when the
    two scores catch DIFFERENT failures.

    Returns:
        corr            -- correlation between the two scores (low = they carry
                           independent information, the precondition for a useful
                           conjunction).
        unique_a        -- fraction of failures ranked risky by A but not B.
        unique_b        -- fraction of failures ranked risky by B but not A.
        joint           -- fraction caught by both.
        complementarity -- unique_a + unique_b: the failures a single score would
                           miss but the pair catches. High = the conjunction is
                           worth the second score.

    This is the analysis that turns "the conjunction had higher coverage" (an
    observation) into "the conjunction helps because these two scores fail on
    disjoint cases" (a mechanism) -- the difference between a number and a
    contribution.
    """
    a = np.asarray(score_a); b = np.asarray(score_b)
    fail = np.asarray(loss).astype(bool)
    corr = float(np.corrcoef(a, b)[0, 1])

    # "flagged risky" = above-median score, among the actual failures
    ma, mb = np.median(a), np.median(b)
    fa, fb = (a > ma), (b > mb)
    F = fail
    n_fail = int(F.sum())
    if n_fail == 0:
        return dict(corr=corr, unique_a=0.0, unique_b=0.0, joint=0.0,
                    complementarity=0.0)
    only_a = int((F & fa & ~fb).sum()) / n_fail
    only_b = int((F & ~fa & fb).sum()) / n_fail
    both = int((F & fa & fb).sum()) / n_fail
    return dict(corr=corr, unique_a=only_a, unique_b=only_b, joint=both,
                complementarity=only_a + only_b)


def best_conjunction(scores, loss, alpha, delta, base_key=None):
    """Search score pairs for the conjunction with the highest certified
    coverage at (alpha, delta). Returns the winning pair and its coverage,
    alongside the best single score, so the gain from conjoining is explicit.
    """
    keys = list(scores)
    single = {k: coverage_at_risk(scores[k], loss, alpha) for k in keys}
    best_single = max(single, key=single.get)
    best_pair, best_cov = None, -1.0
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            res = ltt_calibrate_conjunctive(scores[ka], scores[kb],
                                            loss, alpha, delta)
            cov = res.get("coverage", 0.0) if isinstance(res, dict) else 0.0
            if cov > best_cov:
                best_cov, best_pair = cov, (ka, kb)
    return dict(best_pair=best_pair, pair_coverage=best_cov,
                best_single=best_single,
                single_coverage=single[best_single],
                gain=best_cov - single[best_single])


def verify_risk_guarantee(scores, loss, alpha, delta, n_trials=200,
                         cal_frac=0.5, seed=0, ltt=None):
    """EMPIRICALLY verify the LTT guarantee: P(R_eval(tau_hat) <= alpha) >= 1-delta.

    The guarantee is a probabilistic statement over the random calibration/
    evaluation split. Computing a single tau_hat does not show it holds; you
    have to REPEAT the split many times and check that the fraction of trials
    whose EVALUATION risk stays under alpha is at least 1 - delta.

    Procedure, per trial:
      1. randomly split the data into calibration and evaluation halves,
      2. calibrate tau_hat on calibration via LTT (so it is certified there),
      3. measure the realised risk R_eval among grasps the rule EXECUTES on the
         held-out evaluation half,
      4. record whether R_eval <= alpha.

    Returns the empirical coverage (fraction of trials meeting the budget), the
    target 1-delta, and whether the guarantee is satisfied empirically. This is
    the experiment that turns "we compute a p-value" into "the guarantee holds
    on this data", and it is the number to report as the core validation.
    """
    if ltt is None:
        ltt = ltt_calibrate_single
    scores = np.asarray(scores); loss = np.asarray(loss).astype(float)
    n = len(loss); n_cal = int(cal_frac * n)
    rng = np.random.default_rng(seed)

    met, eval_risks, coverages, n_valid = 0, [], [], 0
    for _ in range(n_trials):
        perm = rng.permutation(n)
        cal, ev = perm[:n_cal], perm[n_cal:]
        res = ltt(scores[cal], loss[cal], alpha, delta)
        if not res.valid:
            continue                       # no certificate this split; skip
        n_valid += 1
        executed = scores[ev] <= res.tau
        if executed.sum() == 0:
            eval_risk = 0.0                # nothing executed -> vacuously safe
        else:
            eval_risk = float(loss[ev][executed].mean())
        eval_risks.append(eval_risk)
        coverages.append(float(executed.mean()))
        if eval_risk <= alpha + 1e-9:
            met += 1

    emp_coverage = met / n_valid if n_valid else 0.0
    return dict(
        alpha=alpha, delta=delta, target=1 - delta,
        n_trials=n_trials, n_certified=n_valid,
        empirical_coverage=emp_coverage,          # frac of trials with R_eval<=alpha
        guarantee_holds=(emp_coverage >= 1 - delta) if n_valid else None,
        mean_eval_risk=float(np.mean(eval_risks)) if eval_risks else None,
        mean_coverage=float(np.mean(coverages)) if coverages else None,
    )


def significance_table(scores: dict, loss: np.ndarray, reference: str | None = None,
                       metric: str = "AURC", n_boot: int = 2000,
                       seed: int = 0, **metric_kw):
    """Compare every score against a reference (default: the cheapest, s_conf).

    s_conf is the right reference because it costs one forward pass. Any score
    requiring extra compute must beat it by a statistically distinguishable
    margin to justify itself; 'numerically better' is not a justification.
    """
    reference = reference or ("s_conf" if "s_conf" in scores else list(scores)[0])
    rows = []
    for key in scores:
        if key == reference:
            continue
        ci = paired_bootstrap_ci(scores[key], scores[reference], loss,
                                 metric=metric, n_boot=n_boot, seed=seed, **metric_kw)
        rows.append({
            "score": key,
            "vs": reference,
            f"delta_{metric}": ci["observed_diff"],
            "ci95_lo": ci["ci_lo"],
            "ci95_hi": ci["ci_hi"],
            "distinguishable": ci["significant"],
        })
    return rows
