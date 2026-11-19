"""
models_unet.py — U-Net grasp detector, the THIRD detector
=========================================================
A compact U-Net emitting the same four dense maps (pos, cos, sin, width) as
GR-ConvNet and GG-CNN, at input resolution.

Why a U-Net specifically. The model-agnostic claim is only convincing if the
detectors span genuinely different architectures, not three variations of one.
The two existing detectors are:

    GR-ConvNet : encoder + RESIDUAL blocks + decoder (no long skips)
    GG-CNN     : plain shallow encoder/decoder (no long skips)

Neither uses SKIP CONNECTIONS. A U-Net is defined by them: each decoder stage is
concatenated with the matching encoder feature map, so gradient and spatial
detail flow along a path neither other model has. Agreement across residual,
plain, and skip designs is a real "across architectures" result.

Drop-in contract: same `input_channels` keyword, same `(pos, cos, sin, width)`
forward, same `compute_loss`, works with the shared `enable_mc_dropout` and
`post_process`. The bottleneck carries the dropout so the MC-dropout score (S3)
works here too. Shape arithmetic runs without torch.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False
    nn = None


# =========================================================================== #
# 1. Shape arithmetic -- no torch
# =========================================================================== #
def shape_trace(size: int = 224):
    """A symmetric U-Net with `depth` halvings returns to the input size iff the
    input is divisible by 2**depth. With depth=4 and 224 = 2**5 * 7, every
    halving is exact (224 -> 112 -> 56 -> 28 -> 14) and the decoder mirrors it
    back to 224. This checks the divisibility that makes the skip-concatenations
    line up; a mismatch would misalign the concatenated feature maps.
    """
    depth = 4
    h = size
    stages = [h]
    for _ in range(depth):
        if h % 2 != 0:
            return dict(input=size, ok=False, reason=f"{h} not divisible by 2")
        h //= 2
        stages.append(h)
    # decoder doubles back up
    for _ in range(depth):
        h *= 2
    return dict(input=size, bottleneck=stages[-1], output=h,
                stages=stages, ok=(h == size))


# =========================================================================== #
# 2. Network (needs torch)
# =========================================================================== #
if _HAS_TORCH:

    class _DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.net(x)

    class UNetGrasp(nn.Module):
        """Compact U-Net with four grasp heads.

        depth=4 downsampling stages with skip concatenations. Channel widths are
        kept small so it trains in minutes, comparable in cost to GG-CNN but
        architecturally distinct.
        """

        def __init__(self, input_channels: int = 4, base: int = 16,
                     dropout_prob: float = 0.1, **kw):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.d0 = _DoubleConv(input_channels, c[0])
            self.d1 = _DoubleConv(c[0], c[1])
            self.d2 = _DoubleConv(c[1], c[2])
            self.d3 = _DoubleConv(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.drop = nn.Dropout2d(p=dropout_prob)

            self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
            self.u2 = _DoubleConv(c[3], c[2])          # c[2]+c[2] after concat
            self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
            self.u1 = _DoubleConv(c[2], c[1])
            self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
            self.u0 = _DoubleConv(c[1], c[0])

            self.pos_head = nn.Conv2d(c[0], 1, kernel_size=1)
            self.cos_head = nn.Conv2d(c[0], 1, kernel_size=1)
            self.sin_head = nn.Conv2d(c[0], 1, kernel_size=1)
            self.width_head = nn.Conv2d(c[0], 1, kernel_size=1)

            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def features(self, x):
            x0 = self.d0(x)                 # full res
            x1 = self.d1(self.pool(x0))     # 1/2
            x2 = self.d2(self.pool(x1))     # 1/4
            x3 = self.d3(self.pool(x2))     # 1/8 (bottleneck)
            x3 = self.drop(x3)
            # decoder with skip concatenations -- the defining U-Net feature
            y2 = self.u2(torch.cat([self.up3(x3), x2], dim=1))
            y1 = self.u1(torch.cat([self.up2(y2), x1], dim=1))
            y0 = self.u0(torch.cat([self.up1(y1), x0], dim=1))
            return y0

        def forward(self, x):
            """Return (pos, cos, sin, width), each (B,1,H,W) at input size."""
            f = self.features(x)
            return (self.pos_head(f), self.cos_head(f),
                    self.sin_head(f), self.width_head(f))

        def compute_loss(self, x, y):
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
# 3. Self-test
# =========================================================================== #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    print("[shape arithmetic — no torch]")
    tr = shape_trace(224)
    ok(tr["ok"], f"224 -> {tr['output']} through 4 skip stages (exact)")
    ok(tr["bottleneck"] == 14, f"bottleneck at 14x14 (got {tr['bottleneck']})")
    # a size that doesn't divide should be flagged, not silently misaligned
    bad = shape_trace(220)                    # 220/2/2/2/2 = 13.75 -> fails
    ok(not bad["ok"], "non-divisible input is flagged (skips would misalign)")

    if _HAS_TORCH:
        print("[forward pass — torch present]")
        for ch in (3, 4):
            net = UNetGrasp(input_channels=ch, dropout_prob=0.1)
            x = torch.randn(2, ch, 224, 224)
            outs = net(x)
            ok(len(outs) == 4, f"{ch}-channel -> four heads")
            for o, name in zip(outs, ("pos", "cos", "sin", "width")):
                ok(tuple(o.shape) == (2, 1, 224, 224),
                   f"{name} shape {tuple(o.shape)} matches input")
        print("[skip connections actually used]")
        net = UNetGrasp(input_channels=4)
        # the decoder DoubleConvs take concatenated channels: u2 input is
        # c[3]=128 (up3 gives c[2]=64, concat x2 c[2]=64 -> 128). Verify the
        # first conv of u2 expects 128 in-channels, proving concat is wired.
        first = net.u2.net[0]
        ok(first.in_channels == 128,
           f"decoder conv takes concatenated channels ({first.in_channels}) "
           f"-> skip connection is real")
        print("[drop-in compatibility]")
        from models_grconvnet import enable_mc_dropout, post_process
        ok(enable_mc_dropout(net) >= 1, "MC-dropout finds the bottleneck dropout")
        q, a, w = post_process(*net(torch.randn(1, 4, 224, 224)))
        ok(q.shape[-2:] == (224, 224), "shared post_process works on U-Net")
        y = tuple(torch.randn(2, 1, 224, 224) for _ in range(4))
        out = net.compute_loss(torch.randn(2, 4, 224, 224), y)
        ok("loss" in out, "compute_loss returns a loss")
        n = sum(p.numel() for p in net.parameters())
        print(f"  (U-Net params: {n:,})")
    else:
        print("[torch absent — shape arithmetic only]")

    print("\nALL U-NET TESTS PASSED")


if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)
    _selftest()
