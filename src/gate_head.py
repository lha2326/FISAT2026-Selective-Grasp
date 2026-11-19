"""
gate_head.py
============
A LEARNED abstention gate for grasp prediction, trained on top of a FROZEN
GR-ConvNet. This is the training component that separates the FISAT paper from
the ICARCV benchmark: ICARCV reads uncertainty out of the backbone; here we
*train a small head* to decide execute-vs-abstain and ask whether learning the
gate beats reading a post-hoc score.

Design decisions and why they matter
------------------------------------
1. THE BACKBONE STAYS FROZEN. We attach a lightweight head to intermediate
   backbone features (or, in the feature-cache path, to a precomputed feature
   vector). Only the head trains. This keeps compute tiny (minutes on Colab)
   and keeps the separation from ICARCV clean: new weights, new objective, but
   no re-training of the perception model.

2. THE HEAD DOES NOT TRAIN ON THE BACKBONE'S TRAINING SPLIT. On train data the
   backbone is over-fit and almost never fails, so failure labels there are
   both rare and unrepresentative. The gate must learn on data where the
   backbone behaves as it will at deployment -> a dedicated `gate-train` split,
   disjoint from the backbone's train/val AND from calibration/eval. See
   `THREE_WAY_SPLIT` note at the bottom.

3. COVERAGE-CONSTRAINED SELECTIVE LOSS (SelectiveNet-style). Minimising failures
   alone has a trivial optimum: abstain on everything. We therefore optimise the
   selective risk subject to a target coverage c, via a quadratic penalty that
   pushes the mean gate activation toward c. We ALSO keep a standard
   classification head as an auxiliary target, which stabilises early training.

4. THE GATE OUTPUT IS STILL JUST A SCORE. Crucially, the trained gate produces a
   scalar in [0,1] (probability of *failure*), which we feed into the SAME
   Learn-then-Test calibration as every post-hoc score. So the learned gate and
   the hand-designed scores compete under one identical risk guarantee -- that
   common evaluation is the fair comparison the paper is built around.

The torch code below is deliberately plain. The numeric core of the selective
loss is separately implemented in numpy and unit-tested at the bottom so its
behaviour is verified even in an environment without torch.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Torch is optional at import time so this file can be inspected/tested without
# a GPU environment. The training code needs it; the numpy reference does not.
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False


# =========================================================================== #
# 1. The head
# =========================================================================== #
if _HAS_TORCH:

    class AbstentionGate(nn.Module):
        """Small head predicting P(failure) from frozen backbone features.

        Two inputs are supported so the head works both when you can run the
        backbone live (feature map input) and when you cached a feature vector
        per sample (vector input):

            forward(feat_map)  with feat_map: (B, C, H, W)  -> pooled internally
            forward(feat_vec)  with feat_vec: (B, D)        -> used directly

        Outputs
        -------
        p_fail   : (B,) predicted failure probability  (the gate SCORE)
        aux_logit: (B,) auxiliary failure logit for the classification loss
        """

        def __init__(self, in_dim: int, hidden: int = 128, p_drop: float = 0.2,
                     from_map: bool = False, map_channels: int | None = None):
            super().__init__()
            self.from_map = from_map
            if from_map:
                c = map_channels or in_dim
                # a 1x1 conv mixes channels, then global stats pool to a vector
                self.mix = nn.Conv2d(c, hidden, kernel_size=1)
                feat_in = hidden * 2                  # mean + max pooled
            else:
                feat_in = in_dim

            self.body = nn.Sequential(
                nn.Linear(feat_in, hidden), nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
            )
            self.gate_head = nn.Linear(hidden // 2, 1)   # selective gate
            self.aux_head = nn.Linear(hidden // 2, 1)    # auxiliary classifier

        def _pool(self, x):
            if not self.from_map:
                return x
            h = F.relu(self.mix(x))                      # (B, hidden, H, W)
            gmean = h.mean(dim=(2, 3))
            gmax = h.amax(dim=(2, 3))
            return torch.cat([gmean, gmax], dim=1)

        def forward(self, x):
            z = self.body(self._pool(x))
            p_fail = torch.sigmoid(self.gate_head(z)).squeeze(-1)
            aux_logit = self.aux_head(z).squeeze(-1)
            return p_fail, aux_logit


# =========================================================================== #
# 2. Coverage-constrained selective loss
# =========================================================================== #
def selective_loss_numpy(p_fail: np.ndarray, fail: np.ndarray,
                         target_coverage: float, lam: float = 32.0,
                         aux_prob: np.ndarray | None = None,
                         alpha_aux: float = 0.5):
    """Reference implementation of the training objective, in numpy.

    We treat `1 - p_fail` as the soft gate activation g (probability of
    EXECUTING). The empirical selective risk is the failure-weighted average
    over executed mass:

        selective_risk = sum(g * fail) / sum(g)

    subject to  mean(g) ~= target_coverage, enforced by a one-sided quadratic
    penalty (only penalise coverage BELOW target, so the model is free to be
    more cautious than asked but is pushed off the abstain-everything optimum).

    The auxiliary term is a plain BCE on failure, which gives the head a dense
    gradient early on before the selective term is meaningful.

    Returns (total, parts_dict) so tests can check each piece.
    """
    p_fail = np.clip(p_fail, 1e-6, 1 - 1e-6)
    g = 1.0 - p_fail                                     # execute mass
    cov = g.mean()
    denom = g.sum() + 1e-8
    sel_risk = float((g * fail).sum() / denom)

    # one-sided: penalise only under-coverage
    cov_gap = max(0.0, target_coverage - cov)
    cov_pen = lam * cov_gap ** 2

    parts = dict(selective_risk=sel_risk, coverage=float(cov),
                 coverage_penalty=float(cov_pen))

    total = sel_risk + cov_pen
    if aux_prob is not None:
        aux_prob = np.clip(aux_prob, 1e-6, 1 - 1e-6)
        bce = float(-(fail * np.log(aux_prob) +
                      (1 - fail) * np.log(1 - aux_prob)).mean())
        parts["aux_bce"] = bce
        total = (1 - alpha_aux) * total + alpha_aux * bce
    parts["total"] = float(total)
    return float(total), parts


if _HAS_TORCH:

    def selective_loss_torch(p_fail, fail, target_coverage: float,
                             lam: float = 32.0, aux_logit=None,
                             alpha_aux: float = 0.5):
        """Torch version of `selective_loss_numpy`, same semantics."""
        p_fail = p_fail.clamp(1e-6, 1 - 1e-6)
        g = 1.0 - p_fail
        cov = g.mean()
        sel_risk = (g * fail).sum() / (g.sum() + 1e-8)
        cov_gap = torch.clamp(target_coverage - cov, min=0.0)
        cov_pen = lam * cov_gap ** 2
        total = sel_risk + cov_pen
        if aux_logit is not None:
            bce = F.binary_cross_entropy_with_logits(aux_logit, fail)
            total = (1 - alpha_aux) * total + alpha_aux * bce
        return total


# =========================================================================== #
# 2b. Split bookkeeping -- torch-free so it can be tested anywhere
# =========================================================================== #
def internal_val_split(train_idx, val_frac: float = 0.2, seed: int = 0):
    """Carve an early-stopping validation slice out of the GATE-TRAIN split only.

    The slice must come from train_idx and nowhere else. If early stopping were
    judged on calibration or evaluation samples, those samples would have
    influenced the gate, they would no longer be exchangeable with deployment
    data, and the Learn-then-Test guarantee built on them would be void -- while
    still printing a perfectly healthy-looking number. Hence a dedicated
    function with its own test rather than an inline permutation.
    """
    train_idx = np.asarray(train_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_val = int(val_frac * len(train_idx))
    return train_idx[perm[n_val:]], train_idx[perm[:n_val]]   # (fit, val)


# =========================================================================== #
# 3. Training loop
# =========================================================================== #
if _HAS_TORCH:

    def train_gate(gate: "AbstentionGate", feats: np.ndarray, fail: np.ndarray,
                   train_idx, target_coverage: float = 0.8, epochs: int = 60,
                   batch_size: int = 256, lr: float = 1e-3, lam: float = 32.0,
                   alpha_aux: float = 0.5, val_frac: float = 0.2,
                   device: str = "cpu", seed: int = 0, verbose: bool = True):
        """Train the gate on the GATE-TRAIN subset of a feature cache.

        feats     : (N, D) feature vectors for ALL samples, OR (N, C, H, W) maps
                    if the head was built with from_map=True.
        fail      : (N,) 0/1 failure indicator for ALL samples.
        train_idx : indices of the gate-train split. REQUIRED, and deliberately
                    positional rather than optional.

        The earlier version of this function took no index argument and split
        `feats` internally, which meant the obvious call trained the gate on
        every sample -- including the calibration and evaluation splits. That
        leak does not raise, does not look wrong in the output, and silently
        voids the risk guarantee, so the subset is now something the caller must
        state rather than something they can forget.

        The early-stopping validation slice is carved from train_idx only.
        Calibration and evaluation are never touched here.
        """
        torch.manual_seed(seed)
        gate = gate.to(device)

        train_idx = np.asarray(train_idx)
        if train_idx.ndim != 1 or len(train_idx) == 0:
            raise ValueError("train_idx must be a non-empty 1-D index array")
        if len(train_idx) == len(fail) and len(fail) > 50:
            raise ValueError(
                "train_idx covers every sample. The gate must train on the "
                "gate-train split only; passing all indices leaks calibration "
                "and evaluation data into gate training and voids the LTT "
                "guarantee. Pass the gate-train indices explicitly.")

        ti, vi = internal_val_split(train_idx, val_frac, seed)

        Xt = torch.as_tensor(feats[ti], dtype=torch.float32, device=device)
        yt = torch.as_tensor(fail[ti], dtype=torch.float32, device=device)
        Xv = torch.as_tensor(feats[vi], dtype=torch.float32, device=device)
        yv = torch.as_tensor(fail[vi], dtype=torch.float32, device=device)

        opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-4)
        best_state, best_val = None, float("inf")

        for ep in range(epochs):
            gate.train()
            perm = torch.randperm(len(ti), device=device)
            for b in range(0, len(ti), batch_size):
                bi = perm[b:b + batch_size]
                p_fail, aux = gate(Xt[bi])
                loss = selective_loss_torch(p_fail, yt[bi], target_coverage,
                                            lam=lam, aux_logit=aux,
                                            alpha_aux=alpha_aux)
                opt.zero_grad(); loss.backward(); opt.step()

            gate.eval()
            with torch.no_grad():
                pv, av = gate(Xv)
                vloss = selective_loss_torch(pv, yv, target_coverage, lam=lam,
                                             aux_logit=av, alpha_aux=alpha_aux)
            if float(vloss) < best_val:
                best_val = float(vloss)
                best_state = {k: v.detach().clone() for k, v in gate.state_dict().items()}
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  train_loss={float(loss):.4f}  "
                      f"val_loss={float(vloss):.4f}")

        if best_state is not None:
            gate.load_state_dict(best_state)
        return gate

    @torch.no_grad()
    def gate_scores(gate: "AbstentionGate", feats: np.ndarray,
                    device: str = "cpu") -> np.ndarray:
        """Emit the learned gate's failure-probability score for a split.

        The returned array plugs straight into selective_grasp.py as another
        column of scores, so LTT calibration and every table treat it exactly
        like S1-S5. That identical treatment is the point: it makes 'learned
        gate vs post-hoc score' a controlled comparison.
        """
        gate = gate.to(device).eval()
        X = torch.as_tensor(feats, dtype=torch.float32, device=device)
        p_fail, _ = gate(X)
        return p_fail.cpu().numpy()


# =========================================================================== #
# 4. numpy self-tests -- runnable without torch
# =========================================================================== #
def _selftest():
    rng = np.random.default_rng(0)
    n = 4000
    # latent difficulty drives both failure and the (noisy) feature the gate sees
    u = rng.random(n)
    fail = (rng.random(n) < u * 0.6).astype(float)
    # a good gate score correlates with u; a useless one is noise
    good = u + 0.15 * rng.normal(size=n)
    good = (good - good.min()) / (good.max() - good.min())
    useless = rng.random(n)

    print("[test] selective loss sanity")
    # abstaining on the riskiest (high p_fail where fail likely) should lower
    # selective risk relative to executing everything
    _, all_exec = selective_loss_numpy(np.zeros(n), fail, 1.0)
    _, gated = selective_loss_numpy(good, fail, 0.7)
    print(f"       exec-all selective risk = {all_exec['selective_risk']:.4f}")
    print(f"       gated    selective risk = {gated['selective_risk']:.4f} "
          f"(coverage {gated['coverage']:.3f})")
    assert gated["selective_risk"] < all_exec["selective_risk"], \
        "a competent gate must reduce selective risk below execute-all"

    print("[test] coverage penalty is one-sided")
    # over-covering (cov > target) must incur NO penalty
    _, over = selective_loss_numpy(np.full(n, 0.01), fail, 0.5)  # cov ~0.99
    assert over["coverage_penalty"] == 0.0, "over-coverage should not be penalised"
    # under-covering must incur a positive penalty
    _, under = selective_loss_numpy(np.full(n, 0.99), fail, 0.9)  # cov ~0.01
    assert under["coverage_penalty"] > 0.0, "under-coverage must be penalised"
    print(f"       over-cover penalty={over['coverage_penalty']:.4f}  "
          f"under-cover penalty={under['coverage_penalty']:.4f}")

    print("[test] useless vs informative gate")
    _, g_good = selective_loss_numpy(good, fail, 0.7)
    _, g_bad = selective_loss_numpy(useless, fail, 0.7)
    print(f"       informative selective risk = {g_good['selective_risk']:.4f}")
    print(f"       useless     selective risk = {g_bad['selective_risk']:.4f}")
    assert g_good["selective_risk"] < g_bad["selective_risk"], \
        "informative features must beat noise"

    print("[test] split discipline")
    train_idx = np.arange(0, 1000)
    fit, val = internal_val_split(train_idx, val_frac=0.2, seed=0)
    assert len(fit) + len(val) == len(train_idx), "split must be a partition"
    assert set(fit).isdisjoint(set(val)), "fit and val must not overlap"
    assert set(fit) | set(val) == set(train_idx), "split must cover train_idx"
    print(f"       fit={len(fit)} val={len(val)} disjoint and covering")

    cal_ev = np.arange(1000, 4000)          # calibration + evaluation indices
    touched = set(fit) | set(val)
    assert touched.isdisjoint(set(cal_ev)), \
        "the gate must never touch calibration or evaluation samples"
    print("       calibration/evaluation untouched by the gate's internal split")

    print("[test] aux BCE present when aux_prob given")
    _, parts = selective_loss_numpy(good, fail, 0.7, aux_prob=good)
    assert "aux_bce" in parts and parts["aux_bce"] > 0
    print(f"       aux_bce={parts['aux_bce']:.4f}")

    print("\nALL TESTS PASSED")


# =========================================================================== #
# 5. Fair comparison helper
# =========================================================================== #
def compare_at_matched_coverage(gate_score: np.ndarray, post_hoc: np.ndarray,
                                loss: np.ndarray, coverage: float,
                                n_min: int = 30):
    """Compare the learned gate and a post-hoc score at IDENTICAL coverage.

    Because the gate's coverage penalty is one-sided, a trained gate is free to
    over-abstain, so comparing the two at their own natural operating points
    would be apples-to-oranges. This fixes coverage for both by thresholding
    each at the same quantile, then reports the selective risk of each. Lower
    selective risk at the same coverage = the better gate. This is the
    comparison the paper's central table should use.
    """
    def sel_risk_at(score):
        tau = np.quantile(score, coverage)
        acc = score <= tau
        if acc.sum() < n_min:
            return float("nan"), float(acc.mean())
        return float(loss[acc].mean()), float(acc.mean())

    r_gate, c_gate = sel_risk_at(gate_score)
    r_post, c_post = sel_risk_at(post_hoc)
    return dict(coverage=coverage,
                gate_selective_risk=r_gate,
                posthoc_selective_risk=r_post,
                gate_wins=bool(r_gate < r_post),
                margin=float(r_post - r_gate))


THREE_WAY_SPLIT = """
Required split discipline (state this in Sec. 5 of the paper):

    backbone train  ->  trains GR-ConvNet          (ICARCV; never reused here)
    backbone val    ->  backbone model selection    (ICARCV; never reused here)
    ------------------------------------------------ test partition below ------
    gate-train      ->  trains AbstentionGate        (this paper)
    calibration     ->  Learn-then-Test threshold    (this paper)
    evaluation      ->  final reported numbers        (this paper)

On Cornell the two boundaries carry DIFFERENT claims and use different modes:

    backbone train/val vs test        -> object (folder) disjoint
        supports: the detector generalises to objects it never saw

    gate-train / calibration / eval   -> image-wise, WITHIN the test partition
        supports: the LTT risk bound is valid on the test pool

The second is image-wise on purpose, not for convenience. Cornell's test
partition spans only a handful of folder groups, so it cannot be divided three
ways by object -- and LTT does not need object novelty, it needs EXCHANGEABILITY
between calibration and evaluation, which a random image split satisfies
exactly. What matters is that gate training never touches the calibration or
evaluation indices; that is enforced by train_gate() requiring train_idx.
"""

if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)
    _selftest()
    print(THREE_WAY_SPLIT)
