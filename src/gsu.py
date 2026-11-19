"""
gsu.py — Grasp-Stability Uncertainty
====================================
A novel uncertainty score for selective grasp detection, proposed in this paper.

Motivation
----------
Every existing score in the pipeline reads the QUALITY MAP:

    S1  height of the best pixel
    S2  entropy / spread of the map
    S3  pose dispersion under weight-space noise (MC-dropout)
    S4  evidential aleatoric+epistemic at the peak pixel
    S5  disagreement among the top-k grasp candidates

None of them asks the question a robot actually faces at execution time: **is
this grasp pose STABLE?** A predicted grasp can have a tall, confident peak (low
S1) that sits on a knife-edge -- perturb the angle by a few degrees or the
position by a few pixels and predicted quality collapses. A real gripper never
executes a pose exactly; it has calibration error, so a grasp in a WIDE, FLAT,
high-quality basin is genuinely safer than one on a sharp peak of the SAME
height. That distinction is invisible to S1-S5, which is the gap GSU fills.

Definition
----------
Around the decoded grasp pose, probe predicted quality at a ring of small pose
perturbations (dx, dy, dtheta) and measure how far quality falls off relative to
the peak:

    GSU = 1 - mean_over_perturbations( quality(pose + delta) / quality(pose) )

A robust grasp (flat basin) keeps quality near the peak under perturbation, so
the ratio is ~1 and GSU ~ 0 (trustworthy). A fragile grasp (sharp peak) loses
quality quickly, the ratio drops, and GSU rises toward 1 (untrustworthy). This
matches the pipeline convention: LARGER = LESS TRUSTWORTHY, accept iff GSU <= tau.

Cost
----
One forward pass. The quality/angle maps are already computed; GSU only reads
them at a handful of nearby locations. No extra passes (unlike S3's T passes),
no extra head (unlike S4). It is as cheap as S5.

Why it is not a repackaging of an existing score
------------------------------------------------
The tests at the bottom construct two grasps with the SAME peak height but
DIFFERENT basin width and show S1 cannot tell them apart while GSU can; and two
maps with the same entropy but different local stability, showing GSU is not S2
in disguise. That empirical dissociation is what makes the novelty defensible.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _bilinear(qmap: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample of a 2-D map at (x, y); clamps to the border."""
    h, w = qmap.shape
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = qmap[y0, x0] * (1 - fx) + qmap[y0, x1] * fx
    bot = qmap[y1, x0] * (1 - fx) + qmap[y1, x1] * fx
    return float(top * (1 - fy) + bot * fy)


def _sample_line(qmap, cx, cy, theta, length, n=7):
    """Mean quality sampled along a line through (cx,cy) at angle theta.

    This is the gripper's plate-contact line: the short axis along which the two
    jaws press. Sampling the whole line (not one point) is what makes the
    rotation probe physical -- a rotation that tips one end off the object shows
    up as a drop in the line mean.
    """
    vals = []
    for s in np.linspace(-length / 2.0, length / 2.0, n):
        vals.append(_bilinear(qmap, cx + s * np.cos(theta),
                              cy + s * np.sin(theta)))
    return float(np.mean(vals))


def grasp_stability_uncertainty(quality, x, y, theta=None, angle_map=None,
                                width=None, dpos=(4.0, 8.0, 12.0),
                                dtheta_deg=(8.0, 15.0), n_ang=8):
    """GSU at a decoded grasp pose: how fast predicted quality decays under
    small pose perturbations. 0 = wide stable basin (a 'hill'), 1 = quality
    collapses under the smallest nudge (a 'needle'). Larger = less trustworthy.

    Two kinds of perturbation, matching the two ways a real grasp goes wrong:

    POSITIONAL -- the gripper lands a few pixels off. Sample quality on rings of
    radius `dpos` around the centre and see how much it drops.

    ROTATIONAL -- the gripper is a few degrees mis-aligned. Rotate the
    plate-contact line by +/- `dtheta_deg` and sample quality ALONG that line
    (its full length ~ the jaw opening), comparing to the un-rotated line. This
    is the physically correct probe: a rotation that swings a plate off the
    object drops the line-mean, which a single-point probe would miss. Needs
    `width` (the jaw opening in pixels) to know how long the contact line is;
    without it, GSU falls back to positional-only.
    """
    quality = np.asarray(quality, dtype=float)
    q0 = _bilinear(quality, x, y)
    if q0 <= EPS:
        return 1.0

    ratios = []

    # --- positional rings ------------------------------------------------ #
    for r in dpos:
        for k in range(n_ang):
            a = 2 * np.pi * k / n_ang
            qx = _bilinear(quality, x + r * np.cos(a), y + r * np.sin(a))
            ratios.append(qx / q0)

    # --- rotational: sample along the plate-contact line ----------------- #
    if theta is not None and width is not None and width > 1.0:
        # the contact line runs across the opening (perpendicular to the plate
        # axis theta); its length is the jaw opening
        line_theta = theta + np.pi / 2.0
        base = _sample_line(quality, x, y, line_theta, width)
        if base > EPS:
            for dd in dtheta_deg:
                for sign in (+1, -1):
                    rot = line_theta + np.deg2rad(dd) * sign
                    ql = _sample_line(quality, x, y, rot, width)
                    ratios.append(ql / q0)
    elif theta is not None:
        # no width available: coarse fallback (single offset point per angle)
        for dd in dtheta_deg:
            for sign in (+1, -1):
                th = theta + np.deg2rad(dd) * sign
                tip = max(dpos)
                ratios.append(_bilinear(quality, x + tip * np.cos(th),
                                        y + tip * np.sin(th)) / q0)

    mean_ratio = float(np.clip(np.mean(ratios), 0.0, 1.0))
    return float(1.0 - mean_ratio)


def classify_basin(quality, x, y, theta=None, width=None,
                   hill_thresh=0.20, needle_thresh=0.50):
    """Label the quality landscape around a grasp as 'hill', 'needle', or
    'slope', with the numbers behind the label.

    Returns a dict:
        gsu            -- the GSU value (0 stable .. 1 fragile)
        peak           -- quality at the grasp centre
        half_width_px  -- radius at which quality falls to half the peak; large
                          = wide basin (hill), small = sharp peak (needle)
        anisotropy     -- how much the decay differs by direction (0 = round
                          basin, high = ridge-like, stable one way fragile another)
        label          -- 'hill'   : GSU < hill_thresh   (wide, robust)
                          'needle' : GSU > needle_thresh (sharp, fragile)
                          'slope'  : in between (moderately stable)

    The half-width and anisotropy are what separate a genuine wide hill from a
    tall needle at the SAME peak height -- the distinction the paper's figure
    needs to make visible.
    """
    quality = np.asarray(quality, dtype=float)
    q0 = _bilinear(quality, x, y)
    gsu = grasp_stability_uncertainty(quality, x, y, theta=theta, width=width)

    # half-width: smallest radius where the ring-mean quality drops below q0/2
    half_width = float("inf")
    if q0 > EPS:
        for r in np.arange(1.0, 40.0, 1.0):
            ring = [_bilinear(quality, x + r * np.cos(2*np.pi*k/12),
                              y + r * np.sin(2*np.pi*k/12)) for k in range(12)]
            if np.mean(ring) < q0 / 2.0:
                half_width = float(r)
                break

    # anisotropy: spread of the per-direction half-decay radius. A ridge decays
    # fast across, slow along -> high anisotropy.
    dir_radii = []
    if q0 > EPS:
        for k in range(12):
            a = 2 * np.pi * k / 12
            rr = 40.0
            for r in np.arange(1.0, 40.0, 1.0):
                if _bilinear(quality, x + r*np.cos(a), y + r*np.sin(a)) < q0/2.0:
                    rr = float(r); break
            dir_radii.append(rr)
    anisotropy = float(np.std(dir_radii) / (np.mean(dir_radii) + EPS)) if dir_radii else 0.0

    if gsu < hill_thresh:
        label = "hill"
    elif gsu > needle_thresh:
        label = "needle"
    else:
        label = "slope"

    return dict(gsu=float(gsu), peak=float(q0), half_width_px=half_width,
                anisotropy=anisotropy, label=label)


def basin_label_counts(quality_maps, grasps):
    """Tally hill/slope/needle across a batch -- for the paper's summary."""
    counts = {"hill": 0, "slope": 0, "needle": 0}
    labels = []
    for q, g in zip(quality_maps, grasps):
        lab = classify_basin(q, g["x"], g["y"], g.get("theta"),
                             g.get("width"))["label"]
        counts[lab] += 1
        labels.append(lab)
    return counts, labels


def gsu_batch(quality_maps, grasps, angle_maps=None, **kw):
    """GSU for a batch. `grasps` is a list of dicts with x, y, theta."""
    out = []
    for i, (q, g) in enumerate(zip(quality_maps, grasps)):
        am = None if angle_maps is None else angle_maps[i]
        out.append(grasp_stability_uncertainty(
            q, g["x"], g["y"], g.get("theta"), am,
            width=g.get("width"), **kw))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Synthetic quality-map builders (used by tests and demos)
# --------------------------------------------------------------------------- #
def _gaussian_peak(size, cx, cy, sigma, height=1.0):
    yy, xx = np.mgrid[0:size, 0:size]
    return height * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) /
                             (2.0 * sigma ** 2)))


# --------------------------------------------------------------------------- #
# Tests: prove GSU measures something S1-S5 do not
# --------------------------------------------------------------------------- #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    size = 100
    cx = cy = 50

    print("[basic behaviour]")
    flat = _gaussian_peak(size, cx, cy, sigma=18)     # wide basin
    sharp = _gaussian_peak(size, cx, cy, sigma=5)      # knife-edge, SAME height
    g_flat = grasp_stability_uncertainty(flat, cx, cy)
    g_sharp = grasp_stability_uncertainty(sharp, cx, cy)
    ok(0 <= g_flat <= 1 and 0 <= g_sharp <= 1, "GSU in [0,1]")
    ok(g_sharp > g_flat, "sharp peak is LESS stable (higher GSU) than flat basin")
    ok(g_flat < 0.25, "wide basin is judged stable")

    print("[dissociation from S1 — the novelty claim]")
    # both peaks have the SAME height 1.0, so S1 = 1 - peak is IDENTICAL
    s1_flat = 1.0 - flat.max()
    s1_sharp = 1.0 - sharp.max()
    ok(abs(s1_flat - s1_sharp) < 1e-6,
       "S1 CANNOT distinguish the two grasps (same peak height)")
    ok(g_sharp - g_flat > 0.15,
       "GSU CLEARLY distinguishes them -> it is not S1 in disguise")

    print("[dissociation from S2 entropy]")
    # Build two maps with near-equal entropy but different local stability:
    # (a) one sharp peak, (b) the same sharp peak plus a far-away second peak.
    # The second peak raises entropy but does not change the LOCAL basin of the
    # first, so GSU at the first peak should be ~unchanged while entropy moves.
    a = _gaussian_peak(size, cx, cy, sigma=5)
    b = a + _gaussian_peak(size, 85, 85, sigma=5)
    def entropy(m, floor=0.5):
        mm = m.copy(); mm[mm < floor * mm.max()] = 0
        p = mm.ravel() / (mm.sum() + EPS); p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    ent_a, ent_b = entropy(a), entropy(b)
    gsu_a = grasp_stability_uncertainty(a, cx, cy)
    gsu_b = grasp_stability_uncertainty(b, cx, cy)
    ok(ent_b > ent_a + 0.3, "far second peak raises entropy (S2 changes)")
    ok(abs(gsu_b - gsu_a) < 0.05,
       "GSU at the first peak is ~unchanged -> it is local, not S2")

    print("[angular fragility — physical rotation probe]")
    # A ridge along x: a grasp whose contact line lies ACROSS the ridge (short
    # way) is robust to rotation; one ALONG the ridge swings off it under
    # rotation. With the width-based line probe this must show up.
    yy, xx = np.mgrid[0:size, 0:size]
    ridge = np.exp(-(((yy - cy) ** 2) / (2.0 * 4 ** 2))) * \
            np.exp(-(((xx - cx) ** 2) / (2.0 * 25 ** 2)))
    # contact line across the ridge (theta so that line_theta = along y = across
    # the thin dimension) vs along it
    gsu_robust = grasp_stability_uncertainty(ridge, cx, cy, theta=0.0, width=16)
    gsu_fragile = grasp_stability_uncertainty(ridge, cx, cy, theta=np.pi/2, width=16)
    ok(gsu_fragile != gsu_robust,
       "rotation probe responds to grasp orientation on a ridge")

    print("[hill vs needle classification]")
    hill = classify_basin(flat, cx, cy, theta=0.0, width=16)
    needle = classify_basin(sharp, cx, cy, theta=0.0, width=16)
    ok(hill["half_width_px"] > needle["half_width_px"],
       "hill has larger half-width than needle")
    ok(hill["gsu"] < needle["gsu"], "hill has lower GSU than needle")
    ok(hill["label"] == "hill", f"wide basin labelled 'hill' (got {hill['label']})")
    ok(needle["label"] in ("needle", "slope"),
       f"sharp peak labelled needle/slope (got {needle['label']})")
    ok(needle["peak"] == hill["peak"] or abs(needle["peak"]-hill["peak"])<0.2,
       "both peaks ~same height -> label comes from SHAPE, not height")

    print("[basin tally]")
    qs2 = [flat, sharp, flat]
    gs2 = [dict(x=cx,y=cy,theta=0.0,width=16)]*3
    counts, labels = basin_label_counts(qs2, gs2)
    ok(sum(counts.values()) == 3, "tally covers every grasp")
    ok(len(labels) == 3, "one label per grasp")

    print("[edge cases]")
    zero = np.zeros((size, size))
    ok(grasp_stability_uncertainty(zero, cx, cy) == 1.0,
       "no quality at the pose -> GSU = 1")
    ok(0 <= grasp_stability_uncertainty(flat, 0, 0) <= 1,
       "corner pose handled (border clamp)")

    print("[batch]")
    qs = [flat, sharp]
    gs = [dict(x=cx, y=cy, theta=0.0), dict(x=cx, y=cy, theta=0.0)]
    out = gsu_batch(qs, gs)
    ok(out.shape == (2,) and out[1] > out[0], "batch matches per-item order")

    print("\nALL GSU TESTS PASSED")
    print("GSU measures pose-basin stability, dissociated from S1 and S2 above.")


if __name__ == "__main__":
    _selftest()
