"""
models/grconvnet.py
===================
GR-ConvNet (Kumra et al., "Antipodal Robotic Grasping using Generative Residual
Convolutional Neural Network"): a fully-convolutional encoder-residual-decoder
emitting four dense maps at input resolution.

    pos   : grasp quality logits (sigmoid at post-processing time)
    cos   : cos(2*theta)
    sin   : sin(2*theta)
    width : gripper opening, rescaled by 150 at decode time

The angle is stored as (cos 2theta, sin 2theta) BECAUSE the gripper angle is
pi-periodic: doubling maps theta and theta+pi onto the same point, so the
network never has to learn a discontinuity that does not physically exist.
Decode with theta = atan2(sin, cos) / 2.

>>> The heads are 1x1 convolutions, and that is load-bearing. <<<
With kernel_size=2, padding=1 they emit H+1 (225 from a 224 input); with
kernel_size=2, padding=0 they emit H-1. Neither crashes -- it silently
misaligns every dense target by a pixel and quietly degrades the metric.
shape_trace() below verifies the whole chain arithmetically, with no torch
required, so this class of bug is caught before any GPU time is spent.
"""

from __future__ import annotations

# torch is imported optionally so shape_trace() and the self-test run in ANY
# environment. The architecture needs torch; the shape arithmetic does not.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False
    nn = None


# =========================================================================== #
# 1. Shape arithmetic -- runs without torch
# =========================================================================== #
def _conv_out(i, k, s=1, p=0, d=1):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def _deconv_out(i, k, s=1, p=0, op=0, d=1):
    return (i - 1) * s - 2 * p + d * (k - 1) + op + 1


def shape_trace(size: int = 224, verbose: bool = False) -> int:
    """Trace the spatial size through the network analytically.

    Every target map, every quality arg-max and every grasp decode assumes the
    output grid lines up with the input grid exactly. Returns the final size so
    a caller can assert it equals `size`.
    """
    h = size
    steps = [("input", h)]
    h = _conv_out(h, 9, 1, 4); steps.append(("conv1  k9 s1 p4", h))
    h = _conv_out(h, 4, 2, 1); steps.append(("conv2  k4 s2 p1", h))
    h = _conv_out(h, 4, 2, 1); steps.append(("conv3  k4 s2 p1", h))
    steps.append(("residual x5", h))
    h = _deconv_out(h, 4, 2, 1); steps.append(("deconv1 k4 s2 p1", h))
    h = _deconv_out(h, 4, 2, 1); steps.append(("deconv2 k4 s2 p1", h))
    h = _conv_out(h, 9, 1, 4); steps.append(("deconv3 k9 s1 p4", h))
    h = _conv_out(h, 1, 1, 0); steps.append(("heads   k1", h))
    if verbose:
        for name, v in steps:
            print(f"    {name:18s} {v}")
    return h


# =========================================================================== #
# 2. Architecture
# =========================================================================== #
if _HAS_TORCH:

    class ResidualBlock(nn.Module):
        """Standard 3x3 residual block; spatial size unchanged."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return y + x

    class GRConvNet(nn.Module):
        """Generative Residual Convolutional Network for grasp detection.

        input_channels : 4 for RGB-D, 3 for RGB only, 1 for depth only
        channel_size   : base width (32 in the reference)
        dropout_prob   : >0 enables decoder dropout, REQUIRED for the S3 score
        n_blocks       : number of residual blocks (5 in the reference)
        """

        def __init__(self, input_channels: int = 4, channel_size: int = 32,
                     dropout_prob: float = 0.1, n_blocks: int = 5):
            super().__init__()
            c = channel_size

            self.conv1 = nn.Conv2d(input_channels, c, 9, stride=1, padding=4)
            self.bn1 = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(c * 2)
            self.conv3 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(c * 4)

            self.res = nn.Sequential(*[ResidualBlock(c * 4, c * 4)
                                       for _ in range(n_blocks)])

            self.deconv1 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
            self.dbn1 = nn.BatchNorm2d(c * 2)
            self.deconv2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
            self.dbn2 = nn.BatchNorm2d(c)
            self.deconv3 = nn.Conv2d(c, c, 9, stride=1, padding=4)

            self.dropout_prob = dropout_prob
            self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

            # 1x1 pointwise heads -- see the module docstring. Do not "improve"
            # these to 2x2 or 3x3 without re-running shape_trace().
            self.pos_head = nn.Conv2d(c, 1, 1)
            self.cos_head = nn.Conv2d(c, 1, 1)
            self.sin_head = nn.Conv2d(c, 1, 1)
            self.width_head = nn.Conv2d(c, 1, 1)

            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def features(self, x):
            """Trunk up to the last decoder feature map.

            Exposed separately so the abstention gate can read a representation
            without a second forward pass.
            """
            y = F.relu(self.bn1(self.conv1(x)))
            y = F.relu(self.bn2(self.conv2(y)))
            y = F.relu(self.bn3(self.conv3(y)))
            y = self.res(y)
            y = F.relu(self.dbn1(self.deconv1(y)))
            y = F.relu(self.dbn2(self.deconv2(y)))
            y = self.deconv3(y)
            if self.dropout is not None:
                y = self.dropout(y)
            return y

        def forward(self, x):
            """Return (pos, cos, sin, width), each (B,1,H,W) at input size."""
            f = self.features(x)
            return (self.pos_head(f), self.cos_head(f),
                    self.sin_head(f), self.width_head(f))

        def compute_loss(self, x, y):
            """Smooth-L1 on all four heads, as in the reference."""
            y_pos, y_cos, y_sin, y_width = y
            pos, cos, sin, width = self(x)
            losses = {
                "p_loss": F.smooth_l1_loss(pos, y_pos),
                "cos_loss": F.smooth_l1_loss(cos, y_cos),
                "sin_loss": F.smooth_l1_loss(sin, y_sin),
                "width_loss": F.smooth_l1_loss(width, y_width),
            }
            return {"loss": sum(losses.values()), "losses": losses,
                    "pred": {"pos": pos, "cos": cos,
                             "sin": sin, "width": width}}


# =========================================================================== #
# 3. MC-dropout helpers
# =========================================================================== #
def enable_mc_dropout(model) -> int:
    """Put ONLY dropout layers into train mode; everything else stays eval.

    model.train() would also un-freeze BatchNorm running statistics, so the MC
    samples would differ for a reason unrelated to model uncertainty -- quietly
    corrupting S3. Returns the number of dropout layers activated so callers can
    assert it is non-zero.
    """
    n = 0
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
            n += 1
    return n


def post_process(pos, cos, sin, width, blur_sigma: float = 2.0):
    """Dense head outputs -> (quality, angle, width) numpy maps."""
    import numpy as np

    # NO sigmoid. The heads are trained to regress the [0,1] quality target
    # directly, so squashing at inference is a train/test mismatch. It does not
    # move the arg-max (sigmoid is monotonic) but it crushes the S1 score:
    # a raw range of 1.00 becomes 0.23 after sigmoid, gutting the very signal
    # the rejection rule thresholds on.
    q = pos.squeeze().detach().cpu().numpy()
    ang = (torch.atan2(sin, cos) / 2.0).squeeze().detach().cpu().numpy()
    w = width.squeeze().detach().cpu().numpy() * 100.0  # matches make_target_maps width_scale
    try:
        from scipy.ndimage import gaussian_filter
        q = gaussian_filter(q, blur_sigma)
        ang = gaussian_filter(ang, blur_sigma)
        w = gaussian_filter(w, blur_sigma)
    except ImportError:
        pass
    return q, ang, w


# =========================================================================== #
# 4. Self-test
# =========================================================================== #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    print("[shape trace @ 224]")
    out = shape_trace(224, verbose=True)
    ok(out == 224, f"224 input -> 224 output (got {out})")

    print("[other input sizes]")
    for size in (128, 240, 300, 320):
        o = shape_trace(size)
        ok(o == size, f"{size} -> {o}")

    print("[head kernel sanity]")
    ok(_conv_out(224, 1, 1, 0) == 224, "1x1 head preserves size")
    ok(_conv_out(224, 2, 1, 1) == 225, "2x2 p1 head would ADD a pixel (the bug)")
    ok(_conv_out(224, 2, 1, 0) == 223, "2x2 p0 head would DROP a pixel")

    print("\nALL SHAPE TESTS PASSED")
    if not _HAS_TORCH:
        print("(torch absent: architecture not instantiated, shapes verified "
              "arithmetically)")


if __name__ == "__main__":
    _selftest()
    if _HAS_TORCH:
        print("\n[live forward pass]")
        net = GRConvNet(input_channels=4, dropout_prob=0.1)
        x = torch.randn(2, 4, 224, 224)
        outs = net(x)
        print("  params:", round(sum(p.numel() for p in net.parameters()) / 1e6, 3), "M")
        print("  head shapes:", [tuple(o.shape) for o in outs])
        assert all(o.shape[-2:] == x.shape[-2:] for o in outs), \
            "heads must emit maps at input resolution"
        n_drop = enable_mc_dropout(net)
        print("  dropout layers for S3:", n_drop)
        assert n_drop > 0, "no dropout -> S3 would be identically zero"
        print("  features:", tuple(net.features(x).shape))
        print("  OK")
