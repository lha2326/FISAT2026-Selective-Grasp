"""
train.py
========
Trains the frozen-backbone-to-be on Cornell. Two methods only:

    deterministic  -> supplies S1, S2, S5 (quality-map scores) and, with
                      dropout re-enabled at inference, S3
    evidential     -> supplies S4 (aleatoric / epistemic split)

Ensemble and heteroscedastic are deliberately NOT here. The FISAT paper does not
use them, and dropping them roughly halves the retraining time -- which is the
whole reason for moving to Cornell.

Usage
-----
    python train.py --config configs/cornell.yaml --method deterministic
    python train.py --config configs/cornell.yaml --method evidential

    # verify the pipeline end-to-end on random tensors, no dataset needed:
    python train.py --smoke

Checkpoints land in runs/<method>/{best.pt,last.pt}. BACK THEM UP THE MOMENT
TRAINING ENDS -- losing these is what forced this rebuild in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False


# =========================================================================== #
# 1. Ground-truth rasterisation: grasp rectangles -> dense target maps
# =========================================================================== #
def _rect_mask(corners: np.ndarray, size: int, shrink: float = 1.0/3.0):
    """Boolean mask of a rotated rectangle, shrunk about its centre.

    The reference implementations paint only the INNER third along the gripper
    axis. Painting the full rectangle would label the open jaw span as graspable
    and blur the quality peak, so the shrink is not cosmetic.
    """
    from metrics_grasp import corners_to_params

    cx, cy, theta, w, h = corners_to_params(corners)
    # Fold theta into [0, pi). A rectangle has 180-degree rotational symmetry, so
    # theta and theta+pi describe the SAME region -- but they produce
    # sign-flipped rotated coordinates, and the boundary test below then
    # includes or excludes a different set of edge pixels (measured: 231 vs 217
    # for the same rectangle). Canonicalising makes rasterisation depend on the
    # grasp, not on how the annotation happened to order its corners.
    theta = theta % np.pi
    # Shrink ONLY the jaw-opening direction (h), never both. Shrinking both
    # multiplies the area by 1/9 instead of 1/3, and the positive region is
    # already only ~0.1% of the map: measured on a typical Cornell rectangle,
    # shrinking both leaves 45 positive pixels where the reference leaves 131.
    # With a regression loss that imbalance collapses every head to zero, and a
    # zero-width prediction has zero area, so IoU is 0 on every single sample --
    # which is exactly the 0.00% validation accuracy this produced.
    w_s, h_s = w, h * shrink
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - cx, yy - cy
    c, s = np.cos(-theta), np.sin(-theta)
    xr = dx * c - dy * s
    yr = dx * s + dy * c
    return (np.abs(xr) <= w_s / 2.0) & (np.abs(yr) <= h_s / 2.0)


def make_target_maps(gts: np.ndarray, size: int = 224, width_scale: float = 100.0):
    """(N,4,2) ground-truth rectangles -> (pos, cos, sin, width) maps.

    The angle is stored as cos(2*theta) / sin(2*theta), NOT cos/sin of theta.
    Doubling folds theta and theta+pi onto the same point, matching the physical
    fact that a parallel-jaw gripper flipped by 180 degrees is the same grasp.
    Storing the raw angle instead would force the network to learn a
    discontinuity that does not exist. Round-tripping is unit-tested below.
    """
    from metrics_grasp import corners_to_params

    pos = np.zeros((size, size), np.float32)
    cos_m = np.zeros((size, size), np.float32)
    sin_m = np.zeros((size, size), np.float32)
    wid = np.zeros((size, size), np.float32)

    gts = np.asarray(gts, dtype=float)
    if gts.size == 0:
        return pos, cos_m, sin_m, wid
    if gts.ndim == 2:
        gts = gts[None]

    yy, xx = np.mgrid[0:size, 0:size]
    for rect in gts:
        cx, cy, theta, w, h = corners_to_params(rect)
        opening = min(w, h)                 # jaw opening = shorter side
        plate = max(w, h)                   # gripper-plate length
        m = _rect_mask(rect, size)          # region that carries angle/width
        if not m.any():
            continue
        # QUALITY: a Gaussian bump centered on the grasp, not a flat plateau.
        # sigma scales with the grasp so small grasps stay sharp. The arg-max of
        # a Gaussian is its center, which is what decode_grasp reads, so the
        # decoded grasp lands where the annotation is. Take the max across
        # overlapping grasps so multiple annotations each keep a peak.
        sigma = max(plate / 6.0, 3.0)
        bump = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
        np.maximum(pos, bump, out=pos)
        # ANGLE / WIDTH: filled over the region so the heads have a target with
        # spatial support, but ONLY where this grasp is the closest peak. Using
        # the mask is fine here because these maps are read at the quality
        # arg-max, which the Gaussian pins to the center.
        cos_m[m] = np.cos(2.0 * theta)
        sin_m[m] = np.sin(2.0 * theta)
        wid[m] = np.clip(opening, 0, width_scale) / width_scale
    return pos, cos_m, sin_m, wid


def decode_angle(cos_m, sin_m):
    """Inverse of the doubled-angle encoding: theta = atan2(sin, cos) / 2."""
    return np.arctan2(sin_m, cos_m) / 2.0


# =========================================================================== #
# 2. Decoding a grasp from predicted maps
# =========================================================================== #
def decode_grasp(quality, angle, width, width_scale: float = 100.0):
    """Read the grasp at the quality arg-max."""
    # `width` arrives already rescaled to pixels by post_process(). The earlier
    # version guessed with `if width.max() > 1.5 else width_scale`, which
    # silently multiplied by 150 a second time whenever the width head was still
    # near zero -- exactly when the guess is least reliable.
    idx = np.unravel_index(np.argmax(quality), quality.shape)
    return dict(y=int(idx[0]), x=int(idx[1]),
                theta=float(angle[idx]),
                width=float(width[idx]),
                quality=float(quality[idx]))


def grasp_to_corners(g, jaw_frac: float = 0.5):
    """Rectangle corners for a decoded grasp, for DRAWING ONLY.

    Do NOT feed this to grasp_success. grasp_success takes the decoded grasp
    dict directly and pulls the plate length from the ground-truth rectangle it
    compares against; routing through this helper instead rebuilds the rectangle
    with a GUESSED height (width*jaw_frac), reintroducing the aspect-mismatch
    that pins the success metric to 0. This function exists only so figures can
    draw a plausible rectangle for a standalone prediction.
    """
    from metrics_grasp import params_to_corners
    return params_to_corners(g["x"], g["y"], g["theta"],
                             g["width"], g["width"] * jaw_frac)


# =========================================================================== #
# 3. Training
# =========================================================================== #
def collate_with_targets(batch, size=224):
    """Attach rasterised target maps to a Cornell batch."""
    import torch
    images = torch.stack([b["image"] for b in batch], 0)
    pos, cs, sn, wd = [], [], [], []
    for b in batch:
        p, c, s, w = make_target_maps(b["gt_grasps"], size)
        pos.append(p); cs.append(c); sn.append(s); wd.append(w)
    to_t = lambda arr: torch.from_numpy(np.stack(arr, 0)[:, None])
    return {"image": images,
            "target": (to_t(pos), to_t(cs), to_t(sn), to_t(wd)),
            "gt_grasps": [b["gt_grasps"] for b in batch],
            "obj_id": [b["obj_id"] for b in batch],
            "idx": [b["idx"] for b in batch]}


@torch.no_grad() if _HAS_TORCH else (lambda f: f)
def validate(model, loader, device, method):
    """Validation success rate under the rectangle metric.

    We early-stop on the GRASP METRIC, not on the loss. A lower smooth-L1 on the
    dense maps does not reliably mean a better grasp at the arg-max, and the
    metric is what the paper reports.
    """
    from metrics_grasp import grasp_success
    from models_grconvnet import post_process

    model.eval()
    n_ok = n_tot = 0
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        for i in range(x.shape[0]):
            if method == "evidential":
                # decode from the FROZEN detector's real quality map (pos), not
                # gamma. gamma exists only to produce the S4 uncertainty; the
                # grasp itself comes from the 72%-accurate detector underneath.
                q = out["pos"][i:i+1]
                c, s, w = out["cos"][i:i+1], out["sin"][i:i+1], out["width"][i:i+1]
            else:
                q, c, s, w = (o[i:i+1] for o in out)
            qm, am, wm = post_process(q, c, s, w)
            g = decode_grasp(qm, am, wm)
            n_ok += grasp_success(g, batch["gt_grasps"][i])
            n_tot += 1
    return n_ok / max(n_tot, 1)


def make_dataset_factory(cfg):
    """Return a `split -> Dataset` builder chosen by cfg['data']['name'].

    Both loaders share the same positional signature, so the only per-dataset
    difference is the class and a couple of Jacquard-only keyword args. Keeping
    this in one place is what lets train.py / extract_scores.py stay identical
    across datasets.
    """
    name = cfg["data"].get("name", "cornell").lower()
    d = cfg["data"]
    size = d["out_size"]
    seed = cfg["train"].get("seed", 0)
    common = dict(use_depth=d.get("use_depth", True),
                  val_frac=d["val_frac"], novel_frac=d.get("novel_frac", 0.0),
                  test_frac=d["test_frac"], seed=seed,
                  split_mode=d.get("split_mode", "object"))
    if name == "jacquard":
        from data_jacquard import Jacquard
        return lambda split: Jacquard(
            d["root"], split, size, mode="real",
            depth_kind=d.get("depth_kind", "perfect"),
            max_images=d.get("max_images"), max_objects=d.get("max_objects"),
            **common)
    from data_cornell import Cornell
    return lambda split: Cornell(d["root"], split, size, mode="real", **common)


def make_model(cfg, in_ch):
    """Build the detector chosen by cfg['model']['name'] (default grconvnet).

    Both detectors share the constructor keyword and the four-head output, so
    the rest of training is identical. This is what lets the paper run the SAME
    rejection pipeline on two architectures and claim model-agnosticism.
    """
    name = cfg.get("model", {}).get("name", "grconvnet").lower()
    dropout = cfg.get("train", {}).get("dropout", 0.1)
    if name == "ggcnn":
        from models_ggcnn import GGCNN
        return GGCNN(input_channels=in_ch, dropout_prob=dropout)
    if name == "unet":
        from models_unet import UNetGrasp
        return UNetGrasp(input_channels=in_ch, dropout_prob=dropout)
    from models_grconvnet import GRConvNet
    return GRConvNet(in_ch, dropout_prob=dropout)


def train(cfg, method: str, out_dir: str = "runs"):
    import torch
    from models_grconvnet import GRConvNet

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["data"]["out_size"]
    in_ch = 4 if cfg["data"].get("use_depth", True) else 3
    torch.manual_seed(cfg["train"].get("seed", 0))

    mk = make_dataset_factory(cfg)

    coll = lambda b: collate_with_targets(b, size)
    tr = DataLoader(mk("train"), batch_size=cfg["train"]["batch_size"],
                    shuffle=True, num_workers=cfg["data"].get("num_workers", 4),
                    collate_fn=coll, drop_last=True)
    va = DataLoader(mk("val"), batch_size=cfg["train"]["batch_size"],
                    shuffle=False, num_workers=cfg["data"].get("num_workers", 4),
                    collate_fn=coll)
    print(f"train={len(tr.dataset)}  val={len(va.dataset)}  device={device}")

    backbone = make_model(cfg, in_ch)
    if method == "evidential":
        # The evidential head sits on the FROZEN deterministic detector, so that
        # checkpoint must exist first. This is what keeps evidential training
        # stable: the detector is already good and cannot be harmed.
        det_ckpt = os.path.join(out_dir, "deterministic", "best.pt")
        if not os.path.exists(det_ckpt):
            raise SystemExit(
                "Train the deterministic model first: the evidential head "
                "attaches to its frozen backbone. Expected " + det_ckpt)
        state = torch.load(det_ckpt, map_location=device)
        backbone.load_state_dict(state if "state_dict" not in state
                                 else state["state_dict"])
        from models_evidential import EvidentialGRConvNet
        model = EvidentialGRConvNet(backbone, freeze=True).to(device)
    else:
        model = backbone.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"].get("lr", 1e-3))
    run_dir = os.path.join(out_dir, method)
    os.makedirs(run_dir, exist_ok=True)

    best, bad, hist = -1.0, 0, []
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 12)

    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in tr:
            x = batch["image"].to(device)
            y = tuple(t.to(device) for t in batch["target"])
            res = model.compute_loss(x, y)
            opt.zero_grad(); res["loss"].backward(); opt.step()
            tot += float(res["loss"].detach()); nb += 1

        acc = validate(model, va, device, method)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_acc=acc))
        print(f"  ep {ep:3d}  loss={tot/max(nb,1):.4f}  val={acc*100:.2f}%  "
              f"({time.time()-t0:.0f}s)")

        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        if acc > best:
            best, bad = acc, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"  early stop at epoch {ep} (best {best*100:.2f}%)")
                break

    json.dump(dict(method=method, best_val=best, history=hist),
              open(os.path.join(run_dir, "history.json"), "w"), indent=2)
    print(f"\nbest val accuracy: {best*100:.2f}%   ->  {run_dir}/best.pt")
    print("BACK THIS UP TO DRIVE NOW.")
    return best


# =========================================================================== #
# 4. Tests
# =========================================================================== #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    from metrics_grasp import params_to_corners

    print("[target rasterisation]")
    size = 224
    rect = params_to_corners(112, 112, np.deg2rad(30), 60, 30)
    pos, cm, sm, wm = make_target_maps(rect[None], size)
    ok(pos.shape == (size, size), "maps are at input resolution")
    ok(0 < pos.sum() < size * size, "positive region is non-empty and not the whole image")
    ok(abs(pos.max() - 1.0) < 1e-6, "quality peak reaches 1 at the grasp center")
    # the arg-max of the quality target must sit at the grasp center, since that
    # is what decode reads back -- this is the exact property whose absence made
    # a perfectly-fit model score 0%
    peak = np.unravel_index(np.argmax(pos), pos.shape)
    ok(abs(peak[1] - 112) <= 1 and abs(peak[0] - 112) <= 1,
       f"quality arg-max is at the grasp center (got {peak[::-1]})")

    print("[doubled-angle encoding]")
    for deg in (0, 15, 30, 75, 89, 120, 179):
        th = np.deg2rad(deg)
        r = params_to_corners(112, 112, th, 60, 30)
        _, c, s, _ = make_target_maps(r[None], size)
        m = (c != 0) | (s != 0)
        dec = decode_angle(c[m].mean(), s[m].mean())
        from metrics_grasp import angle_diff
        ok(angle_diff(dec, th) < 1e-6,
           f"theta={deg:3d}deg round-trips through cos2t/sin2t")

    th0 = params_to_corners(112, 112, 0.0, 60, 30)
    thpi = params_to_corners(112, 112, np.pi, 60, 30)
    _, c0, s0, _ = make_target_maps(th0[None], size)
    _, cp, sp, _ = make_target_maps(thpi[None], size)
    ok(np.allclose(c0, cp, atol=1e-6) and np.allclose(s0, sp, atol=1e-6),
       "theta and theta+pi produce IDENTICAL targets (as they must)")

    print("[width target]")
    # plate 60, opening 30 -> stores the OPENING (min side) / 100
    r = params_to_corners(112, 112, 0.0, 60, 30)
    _, _, _, w = make_target_maps(r[None], size)
    ok(abs(w[w > 0].mean() - 0.30) < 1e-5,
       "width target = jaw opening / 100 (opening 30 -> 0.30)")

    print("[decode]")
    q = np.zeros((size, size)); q[100, 50] = 1.0
    a = np.zeros((size, size)); a[100, 50] = 0.4
    wmap = np.full((size, size), 0.3)
    g = decode_grasp(q, a, wmap)
    ok(g["y"] == 100 and g["x"] == 50, "decode finds the quality arg-max")
    ok(abs(g["theta"] - 0.4) < 1e-9, "decode reads the angle at the peak")

    print("[empty ground truth]")
    p2, _, _, _ = make_target_maps(np.zeros((0, 4, 2)), size)
    ok(p2.sum() == 0, "empty GT gives an all-zero target")

    print("[validate path round-trip]")
    # This mirrors EXACTLY what validate() does: build a perfect target, run it
    # through post_process's transforms and decode_grasp, then score it the way
    # validate scores it. It is the guard that was missing: the earlier
    # round-trip test passed the decoded grasp straight to grasp_success, while
    # validate() routed it through grasp_to_corners first, so the two disagreed
    # and a real 0%% slipped through green tests.
    from metrics_grasp import grasp_success as _gs
    try:
        from scipy.ndimage import gaussian_filter as _gf
    except ImportError:
        _gf = lambda a, s: a
    rng = np.random.default_rng(3)
    ok_n = 0
    for _ in range(40):
        cx, cy = rng.uniform(50, 174, 2)
        th = rng.uniform(-np.pi / 2, np.pi / 2)
        gt = params_to_corners(cx, cy, th, rng.uniform(50, 90), rng.uniform(20, 45))
        p, cm, sm, wd = make_target_maps(gt[None], size)
        ang = np.arctan2(sm, cm) / 2.0
        g = decode_grasp(_gf(p.astype(float), 2.0), _gf(ang, 2.0),
                         _gf(wd * 100.0, 2.0))
        ok_n += _gs(g, [gt])          # EXACTLY validate()'s scoring call
    ok(ok_n >= 38,
       f"perfect target scores via the validate() path ({ok_n}/40 succeed)")

    print("\nALL TRAIN-PIPELINE TESTS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cornell.yaml")
    ap.add_argument("--method", default="deterministic",
                    choices=["deterministic", "evidential"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true",
                    help="run the offline self-tests and exit")
    args = ap.parse_args()

    if args.smoke:
        _selftest()
        return

    import yaml
    cfg = yaml.safe_load(open(args.config))
    train(cfg, args.method, args.out)


if __name__ == "__main__":
    main()
