"""
data_jacquard.py
================
Jacquard V2 loader with the SAME constructor signature as the Cornell loader, so
train.py / extract_scores.py use it unchanged: swap the import and the config.

Jacquard V2 layout (from the official release, github.com/lqh12345/Jacquard_V2):

    JacquardV2_Dataset_<k>/
        <object_hash>/
            <view>_<hash>_grasps.txt        grasp annotations, one per line
            <view>_<hash>_RGB.png           RGB image (1024x1024)
            <view>_<hash>_perfect_depth.tiff   simulated depth (used here)
            <view>_<hash>_stereo_depth.tiff    generated stereo depth
            <view>_<hash>_mask.png          object segmentation mask

Two structural facts drive the design:

1. THE OBJECT ID IS IN THE PATH. Unlike Cornell, every file name carries the
   object hash, so an object-disjoint split is exact, not approximated by folder
   number. Different VIEWS of the same object must never straddle the split, so
   grouping is by hash, and all four views of an object move together.

2. GRASPS ARE ONE PER LINE as `x;y;theta;opening_width;jaw_size` (semicolons,
   theta in DEGREES, coordinates in the ORIGINAL image pixels). We convert each
   to the (4,2) corner array the rest of the pipeline expects, in the resized
   224x224 frame. jaw_size is the plate length; opening_width is the jaw
   opening -- the same opening/plate distinction the metric relies on.

Jacquard is ~51k images. `max_images` and `max_objects` let you work on a
manageable slice while wiring things up; leave them None for the full run.
"""

from __future__ import annotations

import glob
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:                                     # pragma: no cover
    _HAS_TORCH = False
    Dataset = object


# --------------------------------------------------------------------------- #
# Backend-agnostic image IO (shared approach with the Cornell loader)
# --------------------------------------------------------------------------- #
def _imread(path: str) -> np.ndarray:
    errors = []
    if path.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            return np.asarray(tifffile.imread(path))
        except Exception as e:
            errors.append(f"tifffile: {e}")
    try:
        from PIL import Image
        return np.asarray(Image.open(path))
    except Exception as e:
        errors.append(f"PIL: {e}")
    try:
        import imageio.v2 as iio
        return np.asarray(iio.imread(path))
    except Exception as e:
        errors.append(f"imageio: {e}")
    raise RuntimeError(f"No image backend could read {path}\n  " +
                       "\n  ".join(errors))


def _resize(arr: np.ndarray, out_size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if (h, w) == (out_size, out_size):
        return arr.astype(np.float32)
    from scipy.ndimage import zoom
    factors = [out_size / h, out_size / w]
    if arr.ndim == 3:
        factors.append(1.0)
    return zoom(arr.astype(np.float32), factors, order=1)


def available_backends() -> dict:
    out = {}
    for name, mod in [("PIL", "PIL.Image"), ("tifffile", "tifffile"),
                      ("imageio", "imageio"), ("scipy", "scipy.ndimage")]:
        try:
            __import__(mod); out[name] = "ok"
        except Exception as e:
            out[name] = f"FAILED: {type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# Grasp-annotation parsing
# --------------------------------------------------------------------------- #
def parse_grasp_line(line: str):
    """One Jacquard grasp line -> (cx, cy, theta_rad, opening, jaw).

    Format: `x;y;theta;opening_width;jaw_size`, theta in degrees. Returns None
    for blank or malformed lines so a stray line can't abort a whole scene.
    """
    parts = line.strip().split(";")
    if len(parts) < 5:
        return None
    try:
        x, y, theta_deg, opening, jaw = (float(p) for p in parts[:5])
    except ValueError:
        return None
    return x, y, np.deg2rad(theta_deg), opening, jaw


def grasp_to_corners(cx, cy, theta, opening, jaw) -> np.ndarray:
    """(cx, cy, theta, opening, jaw) -> (4,2) corner array in (x, y).

    The rectangle's long edge is the plate length (jaw), the short edge is the
    opening. Corner order matches params_to_corners so corners_to_params
    recovers the same theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    hw, hh = jaw / 2.0, opening / 2.0          # half plate, half opening
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return base @ R.T + np.array([cx, cy], dtype=float)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class Jacquard(Dataset):
    """Jacquard V2, drop-in for the Cornell loader's signature.

    Signature intentionally matches Cornell(root, split, out_size, use_depth,
    mode, val_frac, novel_frac, test_frac, seed, split_mode) so train.py needs
    only its import changed.
    """

    def __init__(self, root, split="train", out_size=224, use_depth=True,
                 mode="real", val_frac=0.1, novel_frac=0.0, test_frac=0.1,
                 seed=0, split_mode="object", depth_kind="perfect",
                 max_images=None, max_objects=None):
        self.root = root
        self.out_size = out_size
        self.use_depth = use_depth
        self.depth_kind = depth_kind          # 'perfect' or 'stereo'

        rgbs = self._scan(root)
        if not rgbs:
            raise FileNotFoundError(
                f"No *_RGB.png found under {root}. Expected Jacquard V2 layout "
                f"JacquardV2_Dataset_*/<hash>/<view>_<hash>_RGB.png")

        # group by object hash so all views of an object share a split
        obj_of = {p: self._hash(p) for p in rgbs}
        objects = sorted(set(obj_of.values()))

        rng = np.random.default_rng(seed)
        if max_objects:
            objects = list(rng.permutation(objects)[:max_objects])
            keep = set(objects)
            rgbs = [p for p in rgbs if obj_of[p] in keep]

        if split_mode == "object":
            sel = self._object_split(objects, rgbs, obj_of, split,
                                     val_frac, test_frac, rng)
        else:
            sel = self._image_split(rgbs, split, val_frac, test_frac, rng)

        if max_images:
            sel = list(sel)[:max_images]

        self.samples = sorted(sel)
        self.obj_ids = [self._hash_to_int(obj_of[p]) for p in self.samples]

    # ---- scanning ------------------------------------------------------- #
    @staticmethod
    def _scan(root):
        # tolerate both the sharded (JacquardV2_Dataset_*) and flat layouts
        hits = glob.glob(os.path.join(root, "**", "*_RGB.png"), recursive=True)
        return sorted(hits)

    @staticmethod
    def _hash(rgb_path):
        # <view>_<hash>_RGB.png -> hash. The hash is everything between the
        # first underscore and the _RGB suffix, since the hash itself has no
        # underscores in the Jacquard naming.
        base = os.path.basename(rgb_path)[:-len("_RGB.png")]
        parts = base.split("_", 1)
        return parts[1] if len(parts) == 2 else base

    _hash_cache: dict = {}

    def _hash_to_int(self, h):
        # stable small int id per object hash, for arrays
        if h not in self._hash_cache:
            self._hash_cache[h] = len(self._hash_cache)
        return self._hash_cache[h]

    # ---- splitting ------------------------------------------------------ #
    def _object_split(self, objects, rgbs, obj_of, split, val_frac,
                      test_frac, rng):
        perm = list(rng.permutation(objects))
        n = len(perm)
        n_test = int(test_frac * n)
        n_val = int(val_frac * n)
        test_o = set(perm[:n_test])
        val_o = set(perm[n_test:n_test + n_val])
        train_o = set(perm[n_test + n_val:])
        pick = {"train": train_o, "val": val_o, "test": test_o}[split]
        return [p for p in rgbs if obj_of[p] in pick]

    def _image_split(self, rgbs, split, val_frac, test_frac, rng):
        perm = list(rng.permutation(rgbs))
        n = len(perm)
        n_test = int(test_frac * n)
        n_val = int(val_frac * n)
        segs = {"test": perm[:n_test],
                "val": perm[n_test:n_test + n_val],
                "train": perm[n_test + n_val:]}
        return segs[split]

    # ---- loading -------------------------------------------------------- #
    def _load_image(self, rgb_path):
        rgb = _imread(rgb_path).astype(np.float32)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, -1)
        rgb = rgb[..., :3] / 255.0
        orig_h, orig_w = rgb.shape[:2]
        rgb = _resize(rgb, self.out_size) / (rgb.max() if rgb.max() > 1 else 1)

        if self.use_depth:
            suffix = f"_{self.depth_kind}_depth.tiff"
            d_path = rgb_path.replace("_RGB.png", suffix)
            if os.path.exists(d_path):
                d = _imread(d_path).astype(np.float32)
                if d.ndim == 3:
                    d = d[..., 0]
                d = _resize(d, self.out_size)
                # normalise depth to [0,1] robustly
                lo, hi = np.percentile(d, 1), np.percentile(d, 99)
                d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
            else:
                d = np.zeros((self.out_size, self.out_size), np.float32)
            img = np.concatenate([rgb, d[..., None]], -1)
        else:
            img = rgb
        return img, (orig_h, orig_w)

    def _load_grasps(self, rgb_path, orig_hw):
        g_path = rgb_path.replace("_RGB.png", "_grasps.txt")
        orig_h, orig_w = orig_hw
        sx = self.out_size / orig_w
        sy = self.out_size / orig_h
        rects = []
        if os.path.exists(g_path):
            for line in open(g_path):
                parsed = parse_grasp_line(line)
                if parsed is None:
                    continue
                cx, cy, theta, opening, jaw = parsed
                # scale center and sizes into the resized frame
                corners = grasp_to_corners(cx * sx, cy * sy, theta,
                                           opening * sy, jaw * sx)
                rects.append(corners)
        if not rects:
            return np.zeros((0, 4, 2), np.float32)
        return np.asarray(rects, np.float32)

    # ---- Dataset API ---------------------------------------------------- #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rgb_path = self.samples[i]
        img, orig_hw = self._load_image(rgb_path)
        grasps = self._load_grasps(rgb_path, orig_hw)
        img_t = (torch.from_numpy(img).permute(2, 0, 1).float()
                 if _HAS_TORCH else img.transpose(2, 0, 1))
        return {"image": img_t, "gt_grasps": grasps,
                "obj_id": self.obj_ids[i], "idx": i}


# --------------------------------------------------------------------------- #
# Self-test (no torch, no dataset needed)
# --------------------------------------------------------------------------- #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    print("[grasp line parsing]")
    p = parse_grasp_line("512.0;480.5;45.0;60.0;120.0")
    ok(p is not None, "well-formed line parses")
    cx, cy, th, opening, jaw = p
    ok(cx == 512.0 and cy == 480.5, "center read correctly")
    ok(abs(th - np.deg2rad(45.0)) < 1e-9, "theta converted deg->rad")
    ok(opening == 60.0 and jaw == 120.0, "opening and jaw read correctly")
    ok(parse_grasp_line("") is None, "blank line -> None")
    ok(parse_grasp_line("1;2;3") is None, "short line -> None")
    ok(parse_grasp_line("a;b;c;d;e") is None, "non-numeric -> None")

    print("[corner construction]")
    from math import isclose
    corners = grasp_to_corners(100, 100, 0.0, 40, 80)
    ok(corners.shape == (4, 2), "four corners")
    # plate (jaw=80) is the long edge, opening=40 the short edge
    w_edge = np.hypot(*(corners[1] - corners[0]))
    h_edge = np.hypot(*(corners[2] - corners[1]))
    ok(isclose(w_edge, 80, abs_tol=1e-6), "long edge == jaw (plate) length")
    ok(isclose(h_edge, 40, abs_tol=1e-6), "short edge == opening")
    ok(isclose(corners.mean(0)[0], 100) and isclose(corners.mean(0)[1], 100),
       "centroid at the grasp center")

    print("[theta round trip through the shared metric]")
    # the metric's corners_to_params must recover the angle we put in
    import importlib.util
    if importlib.util.find_spec is not None:
        try:
            from metrics_grasp import corners_to_params, angle_diff
            for deg in (0, 30, 75, 120, 179):
                c = grasp_to_corners(100, 100, np.deg2rad(deg), 40, 80)
                _, _, rec_th, _, _ = corners_to_params(c)
                ok(angle_diff(rec_th, np.deg2rad(deg)) < 1e-6,
                   f"theta={deg}deg recovered by corners_to_params")
        except ImportError:
            print("  (metrics_grasp not importable here; skipped)")

    print("[hash extraction]")
    j = Jacquard.__new__(Jacquard)
    h = Jacquard._hash("/x/JacquardV2_Dataset_0/abc123def/0_abc123def_RGB.png")
    ok(h == "abc123def", f"object hash parsed from path (got {h})")
    h2 = Jacquard._hash("/x/2_abc123def_RGB.png")
    ok(h2 == "abc123def", "hash parsed regardless of view index")

    print("\nALL JACQUARD LOADER TESTS PASSED")


if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)
    print("backends:", available_backends())
    _selftest()
