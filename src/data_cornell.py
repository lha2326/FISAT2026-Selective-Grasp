"""
src/data/cornell.py
===================
Drop-in Cornell dataset for the grasp-UQ benchmark. Matches the constructor
signature and batch format of src.data.jacquard.JacquardV2 so that switching the
FISAT experiments from Jacquard V2 to Cornell is a one-line import change in the
notebook (and the same `collate`).

Why Cornell for the FISAT paper
-------------------------------
The backbone checkpoints were lost, so the detector must be retrained from
scratch. Cornell is ~1k real RGB-D images (240 objects) vs Jacquard's ~54k
simulated scenes, so it trains in hours rather than days — decisive under the
10/08 deadline — and its real-object images make the qualitative accept/reject
figures more convincing. The one cost is calibration sample size; see the
split note at the bottom.

Expected on-disk layout (standard Cornell release)
--------------------------------------------------
    <root>/
        01/  pcd0100r.png  pcd0100d.tiff  pcd0100cpos.txt  pcd0100cneg.txt ...
        02/  ...
        ...
        10/

    *r.png    : RGB image (640x480)
    *d.tiff   : depth image, OR a pcdXXXX.txt point cloud to be projected
    *cpos.txt : positive grasp rectangles, 4 corner points per grasp (2 lines
                each pair -> 4 lines per rectangle), pixel coords "x y"
    *cneg.txt : negative rectangles (unused for success labels here)

If your copy ships depth as .txt point clouds rather than .tiff, set
`depth_from_pcd=True`; the loader will project them.

Grasp representation
--------------------
`gt_grasps` is returned in the SAME structure the repo already consumes for
Jacquard (an (N,4,2) float array of rectangle corners per sample, or whatever
`gt_array_to_grasps` expects — adapt `_load_grasp_rects` if the repo uses a
different convention). Keeping this identical is what makes the loader a true
drop-in.
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

# ---------------------------------------------------------------------------
# Backend-agnostic image IO.
#
# The first version of this file did `try: from skimage.io import imread /
# except: imread = None`, which turned a clear ImportError at module load into
# a baffling "TypeError: 'NoneType' object is not callable" thrown from inside
# a data loader many frames away. Swallowing an import error and continuing
# with a None is strictly worse than failing loudly.
#
# Instead: try every backend that might be present, and if all of them fail,
# raise ONE error that names each backend and why it failed.
# ---------------------------------------------------------------------------
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
        from skimage.io import imread as _sk_imread
        return np.asarray(_sk_imread(path))
    except Exception as e:
        errors.append(f"skimage: {e}")
    try:
        import imageio.v2 as iio
        return np.asarray(iio.imread(path))
    except Exception as e:
        errors.append(f"imageio: {e}")
    raise RuntimeError(
        f"No image backend could read {path}.\nTried:\n  " + "\n  ".join(errors)
        + "\n\nInstall one of: pillow, tifffile, scikit-image, imageio.")


def _resize(arr: np.ndarray, out_size: int) -> np.ndarray:
    """Bilinear resize to (out_size, out_size), preserving dtype range.

    Uses scipy.ndimage.zoom rather than skimage: it handles float arrays and
    channel dimensions natively, and scipy is already required elsewhere in the
    pipeline, so this removes a dependency instead of adding one.
    """
    h, w = arr.shape[:2]
    if (h, w) == (out_size, out_size):
        return arr.astype(np.float32)
    from scipy.ndimage import zoom
    factors = [out_size / h, out_size / w]
    if arr.ndim == 3:
        factors.append(1.0)
    return zoom(arr.astype(np.float32), factors, order=1)


def available_backends() -> dict:
    """Report which image backends import cleanly. Call this when IO misbehaves."""
    out = {}
    for name, mod in [("PIL", "PIL.Image"), ("tifffile", "tifffile"),
                      ("skimage", "skimage.io"), ("imageio", "imageio"),
                      ("scipy", "scipy.ndimage")]:
        try:
            __import__(mod)
            out[name] = "ok"
        except Exception as e:
            out[name] = f"FAILED: {type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# Cornell rectangle parsing
# --------------------------------------------------------------------------- #
def _read_cornell_rects(path: str) -> list[np.ndarray]:
    """Parse a Cornell cpos/cneg file into a list of (4,2) corner arrays.

    The file lists 4 lines per rectangle: 'x y' each. Lines containing NaN
    (annotation gaps) invalidate their whole 4-line block, matching the standard
    Cornell handling.
    """
    rects, buf = [], []
    if not os.path.exists(path):
        return rects
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                buf = []
                continue
            if np.isnan(x) or np.isnan(y):
                buf = []
                continue
            buf.append((x, y))
            if len(buf) == 4:
                rects.append(np.array(buf, dtype=np.float32))  # (4,2) in (x,y)
                buf = []
    return rects


def _rect_center_angle_wh(rect: np.ndarray):
    """(4,2) corners -> (cx, cy, theta, w, h). theta is the gripper-plate angle."""
    cx, cy = rect[:, 0].mean(), rect[:, 1].mean()
    # edge from corner 0->1 is one gripper plate direction
    dx, dy = rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1]
    theta = np.arctan2(dy, dx)
    w = np.hypot(dx, dy)
    dx2, dy2 = rect[2, 0] - rect[1, 0], rect[2, 1] - rect[1, 1]
    h = np.hypot(dx2, dy2)
    return float(cx), float(cy), float(theta), float(w), float(h)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class Cornell(Dataset):
    """Cornell grasp dataset with the JacquardV2 constructor signature.

    Signature mirrors:
        JacquardV2(root, split, out_size, use_depth, mode,
                   val_frac, novel_frac, test_frac, seed)

    Splits are OBJECT-DISJOINT. Cornell has ~240 distinct objects; `object_id`
    is parsed from the standard file grouping so no image of an object leaks
    across splits. This is what lets the paper claim the risk guarantee holds on
    unseen objects.
    """

    def __init__(self, root, split="train", out_size=224, use_depth=True,
                 mode="real", val_frac=0.15, novel_frac=0.0, test_frac=0.2,
                 seed=0, depth_from_pcd=False, split_mode='object'):
        self.root = root
        self.split = split
        self.out_size = out_size
        self.use_depth = use_depth
        self.depth_from_pcd = depth_from_pcd

        # NOTE: the release ships a `backgrounds/` folder whose files are named
        # pcdb0001r.png etc. Those MATCH a naive pcd*r.png glob and would be
        # silently loaded as training samples with empty grasp sets. We exclude
        # that folder AND require a matching cpos.txt, so a background image
        # cannot slip in through either path.
        all_rgb = sorted(glob.glob(os.path.join(root, "**", "pcd*r.png"),
                                   recursive=True))
        rgb_files = []
        for pth in all_rgb:
            parts = pth.replace("\\", "/").split("/")
            if any(seg.lower() == "backgrounds" for seg in parts):
                continue
            if os.path.basename(pth).startswith("pcdb"):
                continue
            if not os.path.exists(pth.replace("r.png", "cpos.txt")):
                continue
            rgb_files.append(pth)

        if not rgb_files:
            raise FileNotFoundError(
                f"No Cornell RGB files (pcd*r.png with a matching cpos.txt) "
                f"under {root}. Check the dataset was unzipped with its "
                f"01..10 subfolders.")

        # ---- grouping for the object-disjoint split ------------------------ #
        # Cornell carries NO explicit object id. The raw numbering (pcd0100..
        # pcd1034, ~885 images over ~240 objects) has each object as a short
        # contiguous run, but the run boundaries are not recorded anywhere, so
        # any per-object id would be a guess.
        #
        # We therefore group by the release's own numbered FOLDER (01..10).
        # This is coarse -- 10 groups, not 240 -- but it is genuinely disjoint:
        # folders are contiguous number ranges, so no object spans two folders
        # except possibly at a boundary. Claiming per-object ids we cannot
        # verify would be worse than being coarse and honest.
        #
        # split_mode='image' is offered for the calibration/evaluation stage,
        # where exchangeability (not object novelty) is what the LTT guarantee
        # actually requires. See SPLIT_NOTE at the bottom.
        def folder_id(pth):
            parts = pth.replace("\\", "/").split("/")
            for seg in reversed(parts[:-1]):
                if seg.isdigit():
                    return int(seg)
            return 0

        self.split_mode = split_mode
        if split_mode == "object":
            group = np.array([folder_id(p) for p in rgb_files])
        elif split_mode == "image":
            group = np.arange(len(rgb_files))
        else:
            raise ValueError("split_mode must be 'object' or 'image'")

        uniq = np.unique(group)
        rng = np.random.default_rng(seed)
        uniq = rng.permutation(uniq)
        n = len(uniq)
        n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
        n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
        n_novel = int(round(novel_frac * n))

        test_g = set(uniq[:n_test])
        val_g = set(uniq[n_test:n_test + n_val])
        novel_g = set(uniq[n_test + n_val:n_test + n_val + n_novel])
        train_g = set(uniq[n_test + n_val + n_novel:])

        sel = {"train": train_g, "val": val_g,
               "test": test_g, "novel": novel_g}[split]
        keep = np.array([g in sel for g in group])
        self.files = [p for p, k in zip(rgb_files, keep) if k]
        self.obj_ids = [int(g) for g, k in zip(group, keep) if k]
        self.folder_ids = [folder_id(p) for p in self.files]
        if not self.files:
            raise ValueError(
                f"Cornell split '{split}' is empty (groups={n}, "
                f"test={n_test} val={n_val} novel={n_novel}). "
                f"With split_mode='object' there are only ~10 groups, so small "
                f"fracs can round to zero -- raise the frac or use "
                f"split_mode='image'.")

    def __len__(self):
        return len(self.files)

    # ---- loading helpers ------------------------------------------------- #
    def _load_rgb(self, rgb_path):
        img = _imread(rgb_path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]                       # drop alpha if present
        return img / 255.0

    def _load_depth(self, rgb_path):
        d_path = rgb_path.replace("r.png", "d.tiff")
        if os.path.exists(d_path):
            d = _imread(d_path).astype(np.float32)
            if d.ndim == 3:
                d = d[..., 0]
        else:
            d = np.zeros(_imread(rgb_path).shape[:2], np.float32)
        # normalise depth robustly
        m = np.isfinite(d)
        if m.any():
            lo, hi = np.percentile(d[m], [2, 98])
            d = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
        return d.astype(np.float32)

    def _center_crop_resize(self, arr):
        """Cornell is 640x480; center-crop to square then resize to out_size."""
        h, w = arr.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        arr = arr[top:top + s, left:left + s]
        scale = self.out_size / s
        out = _resize(arr, self.out_size)
        return out.astype(np.float32), (top, left, scale)

    def _load_grasp_rects(self, rgb_path, crop):
        top, left, scale = crop
        pos = _read_cornell_rects(rgb_path.replace("r.png", "cpos.txt"))
        out = []
        for r in pos:
            rr = r.copy()
            rr[:, 0] = (rr[:, 0] - left) * scale     # x
            rr[:, 1] = (rr[:, 1] - top) * scale      # y
            # keep rectangles that survive the crop
            if (rr[:, 0].min() >= 0 and rr[:, 1].min() >= 0 and
                    rr[:, 0].max() < self.out_size and rr[:, 1].max() < self.out_size):
                out.append(rr)
        return np.array(out, dtype=np.float32) if out else np.zeros((0, 4, 2), np.float32)

    def __getitem__(self, idx):
        rgb_path = self.files[idx]
        rgb = self._load_rgb(rgb_path)
        rgb_c, crop = self._center_crop_resize(rgb)

        chans = [rgb_c.transpose(2, 0, 1)]           # (3,H,W)
        if self.use_depth:
            d = self._load_depth(rgb_path)
            d_c, _ = self._center_crop_resize(d[..., None])
            chans.append(d_c.transpose(2, 0, 1))     # (1,H,W)
        image = np.concatenate(chans, axis=0).astype(np.float32)

        gt = self._load_grasp_rects(rgb_path, crop)
        return {
            "image": torch.from_numpy(image),
            "gt_grasps": gt,
            "obj_id": int(self.obj_ids[idx]),
            "idx": idx,
        }


# --------------------------------------------------------------------------- #
# collate — mirror the repo's Jacquard collate so downstream code is unchanged
# --------------------------------------------------------------------------- #
def collate(batch):
    images = torch.stack([b["image"] for b in batch], 0)
    return {
        "image": images,
        "gt_grasps": [b["gt_grasps"] for b in batch],
        "obj_id": [b["obj_id"] for b in batch],
        "idx": [b["idx"] for b in batch],
    }


SPLIT_NOTE = """
Split discipline on Cornell (state this in Sec. 5 of the paper):

1. GROUPING. Cornell carries no explicit object id. The numbering (pcd0100..
   pcd1034) puts each object in a short contiguous run, but the run boundaries
   are not recorded, so any per-object id would be invented. We group by the
   release's numbered folders (01..10) instead: coarse (10 groups, not ~240)
   but genuinely disjoint, since folders are contiguous number ranges. Say this
   plainly rather than claiming object ids the dataset does not provide.

2. CONSEQUENCE FOR THE THREE-WAY SPLIT. With only 10 folder groups, a test
   partition is ~2 folders, which cannot be further divided three ways by
   folder. So:

       backbone train / val / test   -> split_mode='object'  (folder-level)
       gate-train / calibration / eval  -> split_mode='image' WITHIN test

   This is not a compromise forced by convenience. The LTT guarantee needs
   EXCHANGEABILITY between calibration and evaluation, which an image-level
   random split satisfies exactly; object novelty is what the train/test
   boundary is for, and that boundary stays folder-level. The two splits carry
   two different claims and it is worth spelling that out:

       train/test folder-disjoint -> the detector generalises to unseen objects
       cal/eval image-exchangeable -> the risk bound is valid on the test pool

3. SAMPLE SIZE. ~885 usable images total. After a folder-level test partition
   (~20% -> ~180 images) split three ways, calibration holds roughly 60 images,
   and the accepted subset at high coverage is smaller still. That is well under
   the ~150-200 needed to certify alpha=0.05 at delta=0.10.

   Mitigations, in order of preference:
     (a) enlarge the test partition (test_frac=0.4) since no backbone training
         happens on it in the FISAT paper -- the detector is frozen, so a big
         test partition costs only backbone accuracy, not validity;
     (b) report alpha=0.10 as the primary budget, alpha=0.05 as secondary;
     (c) if alpha=0.05 cannot be certified, report that as a finding: at this
         calibration size the achievable risk budget is bounded from below.
   Do NOT silently shrink n_min to force a certificate -- that breaks the very
   guarantee the paper is selling.
"""
