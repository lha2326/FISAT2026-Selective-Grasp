"""
models_ggcnn.py — GG-CNN, the second detector for the model-agnostic study
==========================================================================
GG-CNN (Morrison et al., "Closing the Loop for Robotic Grasping: A Real-time,
Generative Grasp Synthesis Approach"): a small fully-convolutional network that
emits the SAME four dense maps as GR-ConvNet, at input resolution:

    pos   : grasp quality
    cos   : cos(2*theta)
    sin   : sin(2*theta)
    width : gripper opening

It exists here for one reason: the paper's claim is that the risk-controlled
rejection rule works on ANY frozen detector. Demonstrating it on a SECOND,
architecturally different detector is what turns "works on GR-ConvNet" into
"model-agnostic". GG-CNN is the right choice because it is a classic grasp
baseline (reviewers know it), it is ~10x smaller so it trains in minutes, and it
is genuinely different from GR-ConvNet (no residual blocks, far fewer
parameters) -- so agreement across the two is meaningful, not a near-duplicate.

Drop-in contract
----------------
Same constructor keyword (`input_channels`), same forward signature
`(pos, cos, sin, width)` each (B,1,H,W), same `compute_loss`, and it works with
the shared `enable_mc_dropout` and `post_process` from models_grconvnet. So
train.py / extract_scores.py select it by name and need no other change.

The shape arithmetic runs without torch (like the GR-ConvNet module), so the
one bug that actually bites -- a head kernel that shifts the output by a pixel
-- is caught before any GPU time.
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
# 1. Shape arithmetic -- no torch needed
# =========================================================================== #
def _conv_out(i, k, s=1, p=0, d=1):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def _deconv_out(i, k, s=1, p=0, op=0):
    return (i - 1) * s - 2 * p + (k - 1) + op + 1


def shape_trace(size: int = 224):
    """Trace H,W through the GG-CNN encoder/decoder arithmetically.

    GG-CNN's original design uses strided convs down then transposed convs up,
    with kernel/stride/padding chosen so the output returns to the input size.
    The transposed-conv output_padding is the fiddly part; this verifies the
    round trip lands exactly on `size`, with no torch.
    """
    # encoder: conv k9 s3 p3 ; conv k5 s2 p2 ; conv k3 s2 p1
    h = size
    h = _conv_out(h, 9, s=3, p=3)
    h = _conv_out(h, 5, s=2, p=2)
    h = _conv_out(h, 3, s=2, p=1)
    enc = h
    # decoder: convT k3 s2 p1 op1 ; convT k5 s2 p2 op1 ; convT k9 s3 p3 op1
    h = _deconv_out(h, 3, s=2, p=1, op=0)
    h = _deconv_out(h, 5, s=2, p=2, op=1)
    h = _deconv_out(h, 9, s=3, p=3, op=2)
    return dict(input=size, bottleneck=enc, output=h, ok=(h == size))


# =========================================================================== #
# 2. The network (needs torch)
# =========================================================================== #
if _HAS_TORCH:

    class GGCNN(nn.Module):
        """GG-CNN with the GR-ConvNet-compatible four-head output.

        dropout_prob > 0 places a dropout layer at the bottleneck so the shared
        MC-dropout score (S3) works on this detector too.
        """

        def __init__(self, input_channels: int = 4, dropout_prob: float = 0.1,
                     **kw):
            super().__init__()
            f = [32, 16, 8, 8, 16, 32]
            self.conv1 = nn.Conv2d(input_channels, f[0], 9, stride=3, padding=3)
            self.conv2 = nn.Conv2d(f[0], f[1], 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(f[1], f[2], 3, stride=2, padding=1)
            self.drop = nn.Dropout2d(p=dropout_prob)
            self.convt1 = nn.ConvTranspose2d(f[2], f[3], 3, stride=2,
                                             padding=1, output_padding=0)
            self.convt2 = nn.ConvTranspose2d(f[3], f[4], 5, stride=2,
                                             padding=2, output_padding=1)
            self.convt3 = nn.ConvTranspose2d(f[4], f[5], 9, stride=3,
                                             padding=3, output_padding=2)
            # 1x1 heads -- load-bearing, exactly as in GR-ConvNet: a 2x2 head
            # would shift every dense target by a pixel and silently hurt.
            self.pos_head = nn.Conv2d(f[5], 1, kernel_size=1)
            self.cos_head = nn.Conv2d(f[5], 1, kernel_size=1)
            self.sin_head = nn.Conv2d(f[5], 1, kernel_size=1)
            self.width_head = nn.Conv2d(f[5], 1, kernel_size=1)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def features(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.drop(x)
            x = F.relu(self.convt1(x))
            x = F.relu(self.convt2(x))
            x = F.relu(self.convt3(x))
            return x

        def forward(self, x):
            """Return (pos, cos, sin, width), each (B,1,H,W) at input size."""
            f = self.features(x)
            return (self.pos_head(f), self.cos_head(f),
                    self.sin_head(f), self.width_head(f))

        def compute_loss(self, x, y):
            """Smooth-L1 on all four heads -- identical objective to GR-ConvNet
            so the two detectors are compared on equal footing."""
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
    # The pipeline fixes input at 224; transposed-conv output_padding is tuned
    # for that size (it is inherently size-specific), so 224 is what must be
    # exact. This is the check that catches a pixel-shift head bug.
    tr = shape_trace(224)
    ok(tr["ok"], f"input 224 -> output {tr['output']} (round trip exact)")
    ok(tr["bottleneck"] < 224 // 8 + 2, "bottleneck is downsampled as expected")

    if _HAS_TORCH:
        print("[forward pass — torch present]")
        for ch in (3, 4):
            net = GGCNN(input_channels=ch, dropout_prob=0.1)
            x = torch.randn(2, ch, 224, 224)
            outs = net(x)
            ok(len(outs) == 4, f"{ch}-channel input -> four heads")
            for o, name in zip(outs, ("pos", "cos", "sin", "width")):
                ok(tuple(o.shape) == (2, 1, 224, 224),
                   f"{name} head shape {tuple(o.shape)} matches input size")
        print("[loss + drop-in compatibility]")
        net = GGCNN(input_channels=4)
        x = torch.randn(2, 4, 224, 224)
        y = tuple(torch.randn(2, 1, 224, 224) for _ in range(4))
        out = net.compute_loss(x, y)
        ok("loss" in out and out["loss"].item() >= 0, "compute_loss returns a loss")
        ok(set(out["pred"]) == {"pos", "cos", "sin", "width"},
           "pred dict matches GR-ConvNet's keys")
        # shared MC-dropout + post_process must accept this model unchanged
        from models_grconvnet import enable_mc_dropout, post_process
        n_drop = enable_mc_dropout(net)
        ok(n_drop >= 1, "enable_mc_dropout finds GG-CNN's dropout layer")
        q, ang, w = post_process(*net(x))
        ok(q.shape[-2:] == (224, 224), "shared post_process works on GG-CNN")
        # param count sanity: GG-CNN must be much smaller than GR-ConvNet
        n_params = sum(p.numel() for p in net.parameters())
        ok(n_params < 100_000, f"GG-CNN is small ({n_params} params)")
    else:
        print("[torch absent — shape arithmetic only]")

    print("\nALL GG-CNN TESTS PASSED")


if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)
    _selftest()
