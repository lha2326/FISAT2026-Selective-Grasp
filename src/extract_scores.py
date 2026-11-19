"""
extract_scores.py
=================
Runs the FROZEN detectors over a Cornell split ONCE and caches everything the
rest of the paper needs into a single .npz:

    idx      (N,)     dataset index, so figures can find the image again
    obj_id   (N,)     folder-level group id, for object-aware splitting
    success  (N,)     1 if the predicted grasp passes the rectangle metric
    s_conf   (N,)     S1  1 - peak quality
    s_ent    (N,)     S2  spatial entropy of the quality map
    s_mc     (N,)     S3  MC-dropout pose dispersion
    s_topk   (N,)     S5  top-k grasp disagreement
    s_ev_al  (N,)     S4  evidential aleatoric      (if --ev-ckpt given)
    s_ev_ep  (N,)     S4  evidential epistemic      (if --ev-ckpt given)
    feat     (N,D)    pooled features for the learned abstention gate

Convention for every score: LARGER = LESS TRUSTWORTHY. Accept iff s <= tau.

This is the ONLY expensive step in the project. Everything downstream reads the
.npz and runs in seconds, so BACK THE .npz UP as soon as it exists.

    python extract_scores.py --config configs/cornell.yaml \
        --ckpt runs/deterministic/best.pt \
        --ev-ckpt runs/evidential/best.pt \
        --split test --out scores.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from train import make_dataset_factory
from metrics_grasp import grasp_success
from models_grconvnet import GRConvNet, enable_mc_dropout, post_process
from train import make_model
from train import decode_grasp, grasp_to_corners

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_state(net, ckpt, device):
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_backbone(ckpt, in_ch, device, dropout=0.1, cfg=None):
    return _load_state(make_model(cfg or {}, in_ch), ckpt, device)


def load_evidential(ckpt, in_ch, device, dropout=0.1, det_ckpt=None, cfg=None):
    from models_evidential import EvidentialGRConvNet
    backbone = make_model(cfg or {}, in_ch)
    if det_ckpt:
        st = torch.load(det_ckpt, map_location=device)
        backbone.load_state_dict(st if "state_dict" not in st else st["state_dict"])
    net = EvidentialGRConvNet(backbone, freeze=True)
    # the saved evidential checkpoint contains the full module (frozen backbone
    # weights included), so loading it restores everything
    return _load_state(net, ckpt, device)


# --------------------------------------------------------------------------- #
# S1 / S2 — quality-map scores (one forward pass)
# --------------------------------------------------------------------------- #
def score_conf(q) -> float:
    """S1: 1 - peak quality. The cheapest score, and the baseline every other
    one must beat to justify its extra compute."""
    return float(1.0 - q.max())


def score_entropy(q, floor_ratio: float = 0.5) -> float:
    """S2: spatial entropy of the quality map above a RELATIVE floor.

    The floor is essential. Without it the near-zero background dominates the
    normalised sum and the entropy becomes a proxy for how much of the frame is
    object, not for grasp ambiguity.
    """
    m = q.copy()
    m[m < floor_ratio * m.max()] = 0.0
    s = m.sum()
    if s <= EPS:
        return float(np.log(m.size))
    p = m.ravel() / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- #
# S5 — top-k grasp disagreement (one forward pass)
# --------------------------------------------------------------------------- #
def score_topk_disagreement(q, ang, w, k: int = 5, nms_radius: int = 20) -> float:
    """Quality-weighted mean pairwise pose distance among top-k local maxima.

    Catches scene ambiguity: several distinct, similarly-good, mutually
    incompatible grasps. Neither peak confidence nor per-pixel variance sees
    this, which is why it is the cheapest useful complement to S1.
    """
    try:
        from skimage.feature import peak_local_max
        peaks = peak_local_max(q, min_distance=nms_radius,
                               threshold_rel=0.5, num_peaks=k)
    except ImportError:
        peaks = np.array([np.unravel_index(np.argmax(q), q.shape)])
    if len(peaks) < 2:
        return 0.0

    diag = float(np.hypot(*q.shape))
    ws = np.array([q[p[0], p[1]] for p in peaks], float)
    ws = ws / (ws.sum() + EPS)

    tot = wsum = 0.0
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            pi, pj = peaks[i], peaks[j]
            d_pos = np.hypot(pi[0] - pj[0], pi[1] - pj[1]) / diag
            d_ang = abs(ang[pi[0], pi[1]] - ang[pj[0], pj[1]]) % np.pi
            d_ang = min(d_ang, np.pi - d_ang) / (np.pi / 2)   # pi-periodic
            d_w = abs(w[pi[0], pi[1]] - w[pj[0], pj[1]]) / 150.0
            wt = ws[i] * ws[j]
            tot += wt * (0.5 * d_pos + 0.35 * d_ang + 0.15 * d_w)
            wsum += wt
    return float(tot / (wsum + EPS))


# --------------------------------------------------------------------------- #
# S3 — MC-dropout dispersion (T forward passes)
# --------------------------------------------------------------------------- #
def circular_std(angles: np.ndarray, period: float = np.pi) -> float:
    """Circular std for pi-periodic gripper angles.

    A plain np.std over angles near the 0/pi wrap reports a huge spread for
    samples that are physically almost identical, which would make S3 fire on
    exactly the scenes where the model is most consistent.
    """
    a = np.asarray(angles, float) * (2 * np.pi / period)
    r = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    r = min(max(r, EPS), 1.0)
    return float(np.sqrt(-2.0 * np.log(r)) * (period / (2 * np.pi)))


def score_mc_dropout(net, x, T: int = 20, lam_theta: float = 0.35,
                     lam_w: float = 0.15) -> float:
    n_layers = enable_mc_dropout(net)
    if n_layers == 0:
        raise RuntimeError(
            "No dropout layers active — S3 would be identically zero and would "
            "silently look like a perfectly confident model. Retrain with "
            "train.dropout > 0 in the config.")
    us, vs, ths, wsv = [], [], [], []
    with torch.no_grad():
        for _ in range(T):
            q, a, w = post_process(*net(x))
            g = decode_grasp(q, a, w)
            us.append(g["x"]); vs.append(g["y"])
            ths.append(g["theta"]); wsv.append(g["width"])
    net.eval()
    diag = float(np.hypot(*q.shape))
    return float(0.5 * (np.std(us) + np.std(vs)) / diag
                 + lam_theta * circular_std(np.asarray(ths))
                 + lam_w * np.std(wsv) / 150.0)


# --------------------------------------------------------------------------- #
# S4 — evidential (one forward pass)
# --------------------------------------------------------------------------- #
def score_evidential(ev_net, x):
    with torch.no_grad():
        out = ev_net(x)
    # select the pixel from the DETECTOR's quality map (pos), then read the
    # evidential parameters there. gamma is not the detector; pos is.
    q = out["pos"].squeeze().cpu().numpy()
    idx = np.unravel_index(np.argmax(q), q.shape)
    nu = float(out["nu"].squeeze().cpu().numpy()[idx])
    al = float(out["alpha"].squeeze().cpu().numpy()[idx])
    be = float(out["beta"].squeeze().cpu().numpy()[idx])
    return (float(be / max(al - 1.0, EPS)),               # aleatoric
            float(be / max(nu * (al - 1.0), EPS)))        # epistemic


# --------------------------------------------------------------------------- #
# Gate features
# --------------------------------------------------------------------------- #
def extract_feature(net, x) -> np.ndarray:
    """Pooled summary of the four heads — the input to the learned gate."""
    with torch.no_grad():
        outs = net(x)
    feats = []
    for m in outs:
        f = m.flatten(1)
        feats += [f.mean(1, keepdim=True), f.amax(1, keepdim=True),
                  f.std(1, keepdim=True)]
    return torch.cat(feats, 1).squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cornell.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ev-ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="scores.npz")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    size = cfg["data"]["out_size"]
    in_ch = 4 if cfg["data"].get("use_depth", True) else 3
    T = cfg["uq"].get("mc_samples", 20)
    drop = cfg["train"].get("dropout", 0.1)

    ds = make_dataset_factory(cfg)(args.split)
    print(f"[info] split='{args.split}'  N={len(ds)}  device={args.device}")

    net = load_backbone(args.ckpt, in_ch, args.device, drop, cfg=cfg)
    ev = load_evidential(args.ev_ckpt, in_ch, args.device, drop,
                         det_ckpt=args.ckpt, cfg=cfg) if args.ev_ckpt else None

    rec = {k: [] for k in ["idx", "obj_id", "success", "s_conf", "s_ent",
                           "s_mc", "s_topk", "s_ev_al", "s_ev_ep"]}
    feats = []

    for i in range(len(ds)):
        sample = ds[i]
        x = sample["image"][None].to(args.device)

        with torch.no_grad():
            raw = net(x)
            q, ang, w = post_process(*raw)
        g = decode_grasp(q, ang, w)

        rec["idx"].append(sample["idx"])
        rec["obj_id"].append(sample["obj_id"])
        rec["success"].append(grasp_success(g, sample["gt_grasps"]))
        rec["s_conf"].append(score_conf(q))
        rec["s_ent"].append(score_entropy(q))
        rec["s_topk"].append(score_topk_disagreement(q, ang, w))
        rec["s_mc"].append(score_mc_dropout(net, x, T=T))
        feats.append(extract_feature(net, x))
        if ev is not None:
            a, e = score_evidential(ev, x)
            rec["s_ev_al"].append(a)
            rec["s_ev_ep"].append(e)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}")

    arrays = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    arrays["feat"] = np.asarray(feats)
    assert len(np.unique(arrays["idx"])) == len(arrays["idx"]), \
        "duplicate indices — is the dataset order non-deterministic?"

    np.savez_compressed(args.out, **arrays)
    print(f"\n[ok] {len(arrays['success'])} samples -> {args.out}")
    print(f"     success rate = {arrays['success'].mean():.4f}")
    print(f"     scores cached: {[k for k in arrays if k.startswith('s_')]}")
    print("\nBACK THIS FILE UP NOW — it is the only expensive artefact here.")


if __name__ == "__main__":
    main()
