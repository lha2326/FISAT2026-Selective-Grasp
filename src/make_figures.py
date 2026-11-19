"""
make_figures.py
===============
Qualitative figures for the paper: which grasps the rule accepts, which it
rejects, and — the panel reviewers actually look for — which ones it accepts and
still gets wrong.

Two figures are produced:

  fig_outcome_grid.pdf   2x2 outcome grid. Rows = accept/reject, columns =
                         success/failure. The (accept, failure) cell is the
                         residual risk; the (reject, success) cell is the
                         coverage you paid. Showing only the two agreeable
                         cells is the most common way this figure gets made and
                         the most common reason it convinces nobody.

  fig_reject_reasons.pdf Rejected samples grouped by WHICH score fired. This is
                         the figure that carries an argument rather than an
                         illustration: if the S1 group, the S2/S5 group and the
                         S4-epistemic group look visibly different, that is the
                         evidence for the claim in Sec. 4.1 that the scores
                         catch different failure modes, and therefore for the
                         conjunctive rule R2.

IMPORTANT — caption wording
---------------------------
Jacquard V2 supplies simulated annotations, not physical trials. Nothing here is
an executed grasp. Caption these as "predicted grasp, evaluated against the
annotated grasp set by the rectangle metric (IoU > 0.25, angle error < 30 deg)".
Writing "successful execution" is a factual error a Jacquard-literate reviewer
will catch immediately.

Usage
-----
    # demo: synthetic renders, runs anywhere
    python make_figures.py --scores scores_demo.npz --demo --out figs/

    # real: wire render_sample() to your dataset + frozen model first
    python make_figures.py --scores scores.npz --tau-conf 0.31 --out figs/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EPS = 1e-12

# Springer LNCS single column is ~122mm; two-column spanning ~ 8.6in wide max.
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_ACCEPT = "#1a7f37"     # green
C_REJECT = "#cf222e"     # red
C_GT = "#0969da"         # blue
C_ALT = "#8250df"        # purple, for top-k alternatives


# --------------------------------------------------------------------------- #
# TODO — wire this to your dataset + frozen model
# --------------------------------------------------------------------------- #
def render_sample(sample_idx: int):
    """Return everything needed to draw one panel.

    Must return a dict:
        image   : (H, W, 3) float in [0,1]  — RGB, or depth colourised
        quality : (H, W)   float in [0,1]   — predicted quality map
        pred    : dict(x, y, theta, width)  — the executed/rejected grasp
        gts     : list of dict(x, y, theta, width) — annotated grasps
        alts    : list of dict(...)          — optional top-k candidates

    Wiring sketch:

        ds = JacquardDataset(root, ...)          # shuffle=False, same order
        x, gt = ds[sample_idx]
        with torch.no_grad(): out = net(x[None].cuda())
        q, ang, w = post_process(*out)
        ...
    """
    raise NotImplementedError("wire render_sample() to your loader + model")


def render_sample_demo(sample_idx: int, rng_seed: int = 0):
    """Synthetic stand-in so the layout can be checked without the model."""
    rng = np.random.default_rng(sample_idx + rng_seed)
    H = W = 224
    yy, xx = np.mgrid[0:H, 0:W]

    img = np.full((H, W, 3), 0.16)
    n_obj = rng.integers(1, 3)
    peaks = []
    for _ in range(n_obj):
        cx, cy = rng.integers(60, W - 60), rng.integers(60, H - 60)
        r = rng.integers(22, 40)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2)))
        img += blob[..., None] * rng.uniform(0.4, 0.8, size=3)
        peaks.append((cx, cy, r))
    img = np.clip(img, 0, 1)

    quality = np.zeros((H, W))
    for cx, cy, r in peaks:
        quality += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (r * 0.6) ** 2)))
    quality = quality / (quality.max() + EPS)

    cx, cy, r = peaks[0]
    pred = dict(x=float(cx), y=float(cy), theta=float(rng.uniform(-np.pi/2, np.pi/2)),
                width=float(r * 1.6))
    gts = [dict(x=float(cx + rng.normal(0, 4)), y=float(cy + rng.normal(0, 4)),
                theta=pred["theta"] + rng.normal(0, 0.25), width=r * 1.7)]
    alts = [dict(x=float(c[0]), y=float(c[1]),
                 theta=float(rng.uniform(-np.pi/2, np.pi/2)), width=c[2] * 1.6)
            for c in peaks[1:]]
    return dict(image=img, quality=quality, pred=pred, gts=gts, alts=alts)


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def grasp_corners(g, jaw_frac: float = 0.45):
    """Four corners of a grasp rectangle centred at (x, y), rotated by theta."""
    w, h = g["width"], g["width"] * jaw_frac
    c, s = np.cos(g["theta"]), np.sin(g["theta"])
    base = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    R = np.array([[c, -s], [s, c]])
    return base @ R.T + np.array([g["x"], g["y"]])


def draw_grasp(ax, g, color, lw=1.1, ls="-", alpha=1.0, jaws=True):
    """Draw a grasp with the two gripper plates emphasised.

    The plate edges are what a reader checks to judge whether the angle is
    right, so they are drawn heavier than the connecting edges.
    """
    p = grasp_corners(g)
    idx = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for k, (i, j) in enumerate(idx):
        heavy = jaws and k in (1, 3)
        ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                color=color, lw=lw * (1.9 if heavy else 1.0),
                ls=ls, alpha=alpha, solid_capstyle="butt")


def draw_panel(ax, s, accepted: bool, success: bool, subtitle: str = "",
               show_quality: bool = True, show_alts: bool = False):
    ax.imshow(s["image"], interpolation="nearest")
    if show_quality:
        ax.imshow(np.ma.masked_less(s["quality"], 0.35), cmap="inferno",
                  alpha=0.38, interpolation="bilinear", vmin=0, vmax=1)

    for g in s["gts"]:
        draw_grasp(ax, g, C_GT, lw=0.8, ls=":", alpha=0.85, jaws=False)
    if show_alts:
        for g in s.get("alts", []):
            draw_grasp(ax, g, C_ALT, lw=0.8, ls="--", alpha=0.8)
    draw_grasp(ax, s["pred"], C_ACCEPT if accepted else C_REJECT, lw=1.3)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(C_ACCEPT if accepted else C_REJECT)
        sp.set_linewidth(1.6)
    if subtitle:
        ax.set_title(subtitle, pad=2, fontsize=6)


# --------------------------------------------------------------------------- #
# Figure 1 — outcome grid
# --------------------------------------------------------------------------- #
CELLS = [
    ("accept_success", True,  True,  "Accepted, correct"),
    ("accept_failure", True,  False, "Accepted, WRONG (residual risk)"),
    ("reject_failure", False, False, "Rejected, would have failed"),
    ("reject_success", False, True,  "Rejected, would have succeeded (cost)"),
]


def pick_examples(mask: np.ndarray, score: np.ndarray, k: int,
                  strategy: str = "spread") -> list[int]:
    """Choose k representative indices from a boolean mask.

    'spread' takes evenly spaced quantiles of the score inside the cell rather
    than the k most extreme cases. Cherry-picking the extremes produces a
    figure that flatters the method and misrepresents the typical case.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    if len(idx) <= k:
        return idx.tolist()
    order = idx[np.argsort(score[idx])]
    if strategy == "spread":
        qs = np.linspace(0.1, 0.9, k)
        return [int(order[int(q * (len(order) - 1))]) for q in qs]
    return order[:k].tolist()


def figure_outcome_grid(scores: dict, loss: np.ndarray, accept: np.ndarray,
                        renderer, out_path: str, k: int = 4, seed: int = 0):
    key = "s_conf" if "s_conf" in scores else list(scores)[0]
    s = scores[key]
    fail = loss.astype(bool)

    fig, axes = plt.subplots(len(CELLS), k, figsize=(k * 1.35, len(CELLS) * 1.5))
    axes = np.atleast_2d(axes)

    for r, (name, acc, succ, label) in enumerate(CELLS):
        mask = (accept == acc) & (fail != succ)
        picks = pick_examples(mask, s, k)
        n_cell = int(mask.sum())
        for c in range(k):
            ax = axes[r, c]
            if c < len(picks):
                draw_panel(ax, renderer(int(picks[c])), acc, succ)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(f"{label}\n(n={n_cell}, {100*n_cell/len(loss):.1f}%)",
                              fontsize=5.5, rotation=0, ha="right", va="center",
                              labelpad=48)

    fig.subplots_adjust(wspace=0.04, hspace=0.12, left=0.30)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ok] {out_path}")


# --------------------------------------------------------------------------- #
# Figure 2 — rejections grouped by triggering score
# --------------------------------------------------------------------------- #
REASON_LABELS = {
    "s_conf":   "S1 low peak quality\n(thin / transparent / odd shape)",
    "s_ent":    "S2 diffuse quality map\n(ambiguous support)",
    "s_topk":   "S5 top-k disagreement\n(competing incompatible grasps)",
    "s_ev_ep":  "S4 high epistemic\n(geometry unlike training set)",
    "s_mc":     "S3 high dropout dispersion\n(unstable pose estimate)",
}


def figure_reject_reasons(scores: dict, loss: np.ndarray, accept: np.ndarray,
                          renderer, out_path: str, k: int = 4,
                          reasons=("s_conf", "s_topk", "s_ev_ep")):
    """Attribute each rejection to the score that is most extreme for it.

    Attribution uses per-score z-scores so that scores on different natural
    scales are comparable; without standardising, whichever score has the
    largest raw spread would claim almost every rejection.
    """
    reasons = [r for r in reasons if r in scores]
    Z = {k_: (scores[k_] - scores[k_].mean()) / (scores[k_].std() + EPS) for k_ in reasons}
    stack = np.stack([Z[k_] for k_ in reasons], axis=1)
    winner = np.argmax(stack, axis=1)

    fig, axes = plt.subplots(len(reasons), k, figsize=(k * 1.35, len(reasons) * 1.5))
    axes = np.atleast_2d(axes)

    for r, key in enumerate(reasons):
        mask = (~accept) & (winner == r)
        picks = pick_examples(mask, scores[key], k)
        for c in range(k):
            ax = axes[r, c]
            if c < len(picks):
                s = renderer(int(picks[c]))
                draw_panel(ax, s, accepted=False, success=False,
                           show_alts=(key in ("s_topk", "s_ent")))
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(f"{REASON_LABELS.get(key, key)}\n(n={int(mask.sum())})",
                              fontsize=5.5, rotation=0, ha="right", va="center",
                              labelpad=52)

    fig.subplots_adjust(wspace=0.04, hspace=0.14, left=0.32)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ok] {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tau-conf", type=float, default=None,
                    help="calibrated threshold on s_conf; if omitted, uses the "
                         "quantile that reproduces --target-coverage")
    ap.add_argument("--target-coverage", type=float, default=0.75)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.scores)
    loss = 1.0 - d["success"].astype(float)
    scores = {k: d[k].astype(float) for k in d.files if k.startswith("s_")}

    key = "s_conf" if "s_conf" in scores else list(scores)[0]
    tau = args.tau_conf if args.tau_conf is not None else \
        float(np.quantile(scores[key], args.target_coverage))
    accept = scores[key] <= tau

    print(f"[info] N={len(loss)}  tau({key})={tau:.4f}  "
          f"coverage={accept.mean():.3f}  "
          f"selective risk={loss[accept].mean():.4f}  "
          f"overall risk={loss.mean():.4f}")

    renderer = render_sample_demo if args.demo else render_sample

    figure_outcome_grid(scores, loss, accept, renderer,
                        os.path.join(args.out, "fig_outcome_grid.pdf"), k=args.k)
    figure_reject_reasons(scores, loss, accept, renderer,
                          os.path.join(args.out, "fig_reject_reasons.pdf"), k=args.k)

    if args.demo:
        print("\n[note] DEMO renders are synthetic. They verify the layout only — "
              "never put them in the paper.")


if __name__ == "__main__":
    main()
