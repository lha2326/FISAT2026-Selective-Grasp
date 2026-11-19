"""
models/evidential.py
====================
Deep Evidential Regression (Amini et al., NeurIPS 2020) on top of GR-ConvNet,
providing the S4 score: a single-pass split of predictive uncertainty into
aleatoric and epistemic parts.

>>> WARNING -- READ BEFORE USING <<<
This is the one module worth cross-checking against your ICARCV implementation.
The NIG loss is easy to get subtly wrong (a misplaced softplus, alpha not
constrained above 1, a sign in the regulariser) in ways that still train and
still emit plausible numbers -- they are just the wrong numbers. Your ICARCV
version is validated by the fact that it produced the "evidential ranks best on
AUSE" result. If the two disagree, trust yours and delete this one.

The maths
---------
The head predicts the four parameters of a Normal-Inverse-Gamma prior over the
target's mean and variance:

    gamma  in R        predicted mean
    nu     > 0         "virtual observations" -- evidence for the mean
    alpha  > 1         evidence for the variance (must exceed 1 or the
                       variance below is undefined)
    beta   > 0         scale

    prediction   = gamma
    aleatoric    = beta / (alpha - 1)              data noise
    epistemic    = beta / (nu * (alpha - 1))       model ignorance

Epistemic carries the 1/nu factor: more evidence for the mean drives it down,
while aleatoric is unaffected by nu. That asymmetry is exactly what makes the
epistemic term the one expected to fire on out-of-distribution geometry, and it
is unit-tested below.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False

EPS = 1e-8


# =========================================================================== #
# 1. numpy reference -- the ground truth for the maths, fully testable
# =========================================================================== #
def nig_nll_numpy(y, gamma, nu, alpha, beta):
    """Negative log-likelihood of the NIG evidential distribution."""
    from scipy.special import gammaln
    y, gamma = np.asarray(y, float), np.asarray(gamma, float)
    nu, alpha, beta = (np.asarray(v, float) for v in (nu, alpha, beta))
    omega = 2.0 * beta * (1.0 + nu)
    return (0.5 * np.log(np.pi / nu)
            - alpha * np.log(omega)
            + (alpha + 0.5) * np.log(nu * (y - gamma) ** 2 + omega)
            + gammaln(alpha) - gammaln(alpha + 0.5))


def nig_regulariser_numpy(y, gamma, nu, alpha):
    """Evidence penalty: scale the error by the total evidence.

    Pushes evidence DOWN where the prediction is wrong, which is what stops the
    network from becoming confidently incorrect.
    """
    return np.abs(np.asarray(y, float) - np.asarray(gamma, float)) * \
        (2.0 * np.asarray(nu, float) + np.asarray(alpha, float))


def nig_uncertainty_numpy(nu, alpha, beta):
    """(aleatoric, epistemic) from the NIG parameters."""
    nu, alpha, beta = (np.asarray(v, float) for v in (nu, alpha, beta))
    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    return aleatoric, epistemic


# =========================================================================== #
# 2. torch head + loss
# =========================================================================== #
if _HAS_TORCH:

    class EvidentialHead(nn.Module):
        """Emits (gamma, nu, alpha, beta) maps with their constraints applied.

        softplus keeps nu and beta strictly positive; alpha additionally gets
        +1 so that alpha > 1 and the variance is defined. Forgetting that +1 is
        the single most common bug in evidential implementations -- it yields
        negative or exploding uncertainties on a subset of pixels.
        """

        def __init__(self, in_channels: int):
            super().__init__()
            # 1x1 pointwise, matching the deterministic heads. kernel_size=2
            # with padding=1 emits H+1 and misaligns the pixel grid.
            self.gamma = nn.Conv2d(in_channels, 1, 1)
            self.nu = nn.Conv2d(in_channels, 1, 1)
            self.alpha = nn.Conv2d(in_channels, 1, 1)
            self.beta = nn.Conv2d(in_channels, 1, 1)

        def forward(self, f):
            gamma = self.gamma(f)
            # clamp the evidence parameters. Without an upper bound a handful of
            # pixels drive nu/alpha up, the NLL runs to large negative values
            # (the -2.4 loss seen in training), and validation accuracy
            # oscillates as the model chases those pixels instead of the grasp.
            nu = torch.clamp(F.softplus(self.nu(f)) + EPS, max=50.0)
            alpha = torch.clamp(F.softplus(self.alpha(f)) + 1.0 + EPS, max=50.0)
            beta = torch.clamp(F.softplus(self.beta(f)) + EPS, min=EPS, max=50.0)
            return gamma, nu, alpha, beta

    def nig_nll(y, gamma, nu, alpha, beta):
        omega = 2.0 * beta * (1.0 + nu)
        return (0.5 * torch.log(np.pi / nu)
                - alpha * torch.log(omega)
                + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
                + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

    def nig_regulariser(y, gamma, nu, alpha):
        return torch.abs(y - gamma) * (2.0 * nu + alpha)

    def evidential_loss(y, gamma, nu, alpha, beta, lam: float = 0.1,
                        reduction: str = "mean"):
        loss = nig_nll(y, gamma, nu, alpha, beta) + \
            lam * nig_regulariser(y, gamma, nu, alpha)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    def nig_uncertainty(nu, alpha, beta):
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return aleatoric, epistemic

    class EvidentialGRConvNet(nn.Module):
        """Evidential uncertainty head on a FROZEN deterministic backbone.

        This is the honest architecture for post-hoc uncertainty on a fixed
        detector, and it is what makes evidential training stable here. An
        earlier version gave the evidential model its OWN trainable backbone and
        let the NIG likelihood model the quality map directly; the likelihood
        then minimised itself by becoming very confident about the vast zero
        background, gamma collapsed toward 0 everywhere, the quality map lost its
        peak, and validation sat at 0% while the loss ran to -1.4.

        Here the backbone is the trained deterministic detector, frozen. It
        supplies the grasp (the 72%-accurate detector), and the evidential head
        -- the ONLY trainable part -- attaches a Normal-Inverse-Gamma
        distribution to the quality target so S4 can be read at the detector's
        selected pixel. The detector cannot be harmed because it does not train,
        and S4 becomes a genuine post-hoc uncertainty rather than a competing,
        unstable detector.

        Pass the SAME grasp maps the detector produces through `forward`, so
        decoding at inference uses the detector's quality peak, not gamma.
        """

        def __init__(self, backbone, channel_size: int = 32, freeze: bool = True):
            super().__init__()
            self.backbone = backbone
            if freeze:
                self.backbone.eval()
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
            self.evidential = EvidentialHead(channel_size)

        def forward(self, x):
            # backbone frozen -> no grad through it; features are detached so the
            # evidential head cannot backprop into (and destabilise) the detector
            if any(p.requires_grad for p in self.backbone.parameters()):
                f = self.backbone.features(x)
            else:
                with torch.no_grad():
                    f = self.backbone.features(x)
                f = f.detach()
            gamma, nu, alpha, beta = self.evidential(f)
            with torch.no_grad():
                pos = self.backbone.pos_head(f)      # detector's real quality map
                cos = self.backbone.cos_head(f)
                sin = self.backbone.sin_head(f)
                width = self.backbone.width_head(f)
            # `pos` is the map used for decoding at validation/inference; `gamma`
            # and the evidence params exist only to produce the S4 score.
            return dict(pos=pos, gamma=gamma, nu=nu, alpha=alpha, beta=beta,
                        cos=cos, sin=sin, width=width)

        def compute_loss(self, x, y, lam: float = 0.1):
            # Only the evidential head trains; it regresses the quality target
            # with the NIG likelihood. The pose heads are frozen, so no loss on
            # them. Class imbalance is handled by weighting the sparse positive
            # region up to match the background mass.
            y_pos = y[0]
            out = self(x)
            per_px = evidential_loss(y_pos, out["gamma"], out["nu"],
                                     out["alpha"], out["beta"], lam=lam,
                                     reduction="none")
            w = 1.0 + 9.0 * (y_pos > 0.1).float()     # upweight the bump 10x
            loss = (per_px * w).sum() / w.sum()
            return {"loss": loss, "losses": {"evidential": loss}, "pred": out}


# =========================================================================== #
# 3. Unit tests on the maths -- run without torch
# =========================================================================== #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    print("[NLL shape]")
    y = 1.0
    base = dict(nu=2.0, alpha=3.0, beta=1.0)
    gammas = np.linspace(-2, 4, 601)
    nlls = np.array([nig_nll_numpy(y, g, **base) for g in gammas])
    ok(np.isfinite(nlls).all(), "NLL finite over a wide range of gamma")
    ok(abs(gammas[np.argmin(nlls)] - y) < 1e-2,
       f"NLL minimised at gamma == y  [argmin={gammas[np.argmin(nlls)]:.3f}]")

    print("[uncertainty decomposition]")
    al, ep = nig_uncertainty_numpy(nu=2.0, alpha=3.0, beta=1.0)
    ok(abs(al - 1.0 / 2.0) < 1e-12, "aleatoric == beta/(alpha-1)")
    ok(abs(ep - 1.0 / 4.0) < 1e-12, "epistemic == beta/(nu(alpha-1))")

    nus = np.array([0.1, 1.0, 10.0, 100.0])
    als, eps = nig_uncertainty_numpy(nus, 3.0, 1.0)
    ok(np.all(np.diff(eps) < 0), "epistemic DECREASES as evidence nu grows")
    # aleatoric = beta/(alpha-1) has no nu term at all, so it broadcasts to a
    # single value no matter how many nu values are passed. That IS the property
    # under test, so assert it after an explicit broadcast rather than indexing.
    als_b = np.broadcast_to(np.asarray(als), nus.shape)
    ok(np.allclose(als_b, als_b.flat[0]), "aleatoric is INDEPENDENT of nu")

    alphas = np.array([1.5, 3.0, 11.0])
    als2, _ = nig_uncertainty_numpy(2.0, alphas, 1.0)
    ok(np.all(np.diff(als2) < 0), "aleatoric decreases as alpha grows")

    print("[regulariser]")
    ok(nig_regulariser_numpy(1.0, 1.0, 2.0, 3.0) == 0.0,
       "regulariser == 0 when the prediction is exact")
    r_small = nig_regulariser_numpy(1.0, 0.5, 2.0, 3.0)
    r_big = nig_regulariser_numpy(1.0, 0.5, 20.0, 3.0)
    ok(r_big > r_small, "more evidence on a WRONG prediction costs more")
    ok(nig_regulariser_numpy(1.0, 2.0, 2.0, 3.0) ==
       nig_regulariser_numpy(1.0, 0.0, 2.0, 3.0),
       "regulariser is symmetric in the error")

    print("[constraint sanity]")
    softplus = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    raw = np.array([-50.0, -5.0, 0.0, 5.0, 50.0])
    ok(np.all(softplus(raw) + EPS > 0), "softplus keeps nu, beta > 0")
    ok(np.all(softplus(raw) + 1.0 + EPS > 1.0),
       "softplus + 1 keeps alpha > 1 (variance defined)")
    al3, ep3 = nig_uncertainty_numpy(softplus(raw) + EPS,
                                     softplus(raw) + 1.0 + EPS, 1.0)
    ok(np.all(np.isfinite(al3)) and np.all(al3 > 0),
       "aleatoric finite and positive at extreme logits")
    ok(np.all(np.isfinite(ep3)) and np.all(ep3 > 0),
       "epistemic finite and positive at extreme logits")

    print("\nALL EVIDENTIAL MATH TESTS PASSED")
    print("Still cross-check against your ICARCV implementation before trusting it.")


if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)
    _selftest()
