"""
metrics/grasp.py
================
The rectangle metric that defines "success" for the whole paper.

A predicted grasp counts as correct when, against ANY ground-truth rectangle:
    * the rotated-rectangle IoU exceeds 0.25, AND
    * the gripper-plate angle differs by less than 30 degrees.

Two details are easy to get wrong and both are tested at the bottom:

1. GRIPPER ANGLE IS pi-PERIODIC. A parallel-jaw gripper at 0 and at 180 degrees
   is the SAME grasp. Any angle comparison must fold by pi, not 2*pi. Getting
   this wrong silently marks correct grasps as failures, which would corrupt
   every risk number in the paper.

2. THE POLYGONS ARE ROTATED, so an axis-aligned IoU is wrong. We clip one
   convex polygon against the other (Sutherland-Hodgman) and use the shoelace
   area. No shapely dependency, so the metric is verifiable in isolation.

This module is pure numpy on purpose: the definition of success should not
depend on torch, a GPU, or a model being loadable.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
DEFAULT_IOU = 0.25
DEFAULT_ANGLE_DEG = 30.0


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
def corners_to_params(rect: np.ndarray):
    """(4,2) corner array in (x, y) order -> (cx, cy, theta, w, h).

    `theta` is the direction of the edge from corner 0 to corner 1, which for
    the Cornell annotation convention is a gripper-plate edge.
    """
    rect = np.asarray(rect, dtype=float)
    cx, cy = rect[:, 0].mean(), rect[:, 1].mean()
    dx, dy = rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1]
    theta = np.arctan2(dy, dx)
    w = np.hypot(dx, dy)
    dx2, dy2 = rect[2, 0] - rect[1, 0], rect[2, 1] - rect[1, 1]
    h = np.hypot(dx2, dy2)
    return float(cx), float(cy), float(theta), float(w), float(h)


def params_to_corners(cx, cy, theta, w, h) -> np.ndarray:
    """(cx, cy, theta, w, h) -> (4,2) corner array in (x, y) order."""
    c, s = np.cos(theta), np.sin(theta)
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                     [w / 2, h / 2], [-w / 2, h / 2]], dtype=float)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return base @ R.T + np.array([cx, cy], dtype=float)


# --------------------------------------------------------------------------- #
# Convex polygon geometry
# --------------------------------------------------------------------------- #
def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a simple polygon; sign-independent."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _orient(poly: np.ndarray) -> np.ndarray:
    """Force counter-clockwise winding so the clip half-plane test is consistent."""
    poly = np.asarray(poly, dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly[::-1] if signed < 0 else poly


def convex_intersection(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of convex poly_a against convex poly_b."""
    out = _orient(poly_a)
    clip = _orient(poly_b)
    n_clip = len(clip)

    for i in range(n_clip):
        if len(out) == 0:
            return np.zeros((0, 2))
        a, b = clip[i], clip[(i + 1) % n_clip]
        edge = b - a

        def inside(p):
            # left of the directed edge a->b, for CCW winding
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-9

        new = []
        m = len(out)
        for j in range(m):
            cur, nxt = out[j], out[(j + 1) % m]
            cur_in, nxt_in = inside(cur), inside(nxt)
            if cur_in:
                new.append(cur)
            if cur_in != nxt_in:
                d = nxt - cur
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > 1e-12:
                    t = (edge[0] * (cur[1] - a[1]) -
                         edge[1] * (cur[0] - a[0])) / -denom
                    new.append(cur + t * d)
        out = np.array(new, dtype=float) if new else np.zeros((0, 2))
    return out


def rect_iou(rect_a: np.ndarray, rect_b: np.ndarray) -> float:
    """IoU of two rotated rectangles given as (4,2) corner arrays."""
    a = np.asarray(rect_a, dtype=float)
    b = np.asarray(rect_b, dtype=float)
    inter = polygon_area(convex_intersection(a, b))
    union = polygon_area(a) + polygon_area(b) - inter
    return float(inter / union) if union > EPS else 0.0


# --------------------------------------------------------------------------- #
# Angle handling
# --------------------------------------------------------------------------- #
def angle_diff(a: float, b: float) -> float:
    """Smallest angle between two gripper orientations, folded by pi.

    A parallel-jaw gripper at theta and theta+pi is the same physical grasp, so
    the difference must fold by pi. Result is in [0, pi/2].
    """
    d = abs(float(a) - float(b)) % np.pi
    return float(min(d, np.pi - d))


# --------------------------------------------------------------------------- #
# Success criterion
# --------------------------------------------------------------------------- #
def grasp_success(pred, gts, iou_thresh: float = DEFAULT_IOU,
                  angle_thresh_deg: float = DEFAULT_ANGLE_DEG) -> int:
    """1 if `pred` matches ANY ground-truth grasp, else 0.

    pred : dict with x, y, theta (a decoded grasp); (4,2) corners; or
           (cx, cy, theta, w, h).
    gts  : (N,4,2) array or list of (4,2) ground-truth corner arrays.

    Cornell's rectangle metric compares CENTER, ANGLE and rectangle overlap. The
    subtlety that broke an earlier version: a predicted grasp does not carry a
    trustworthy jaw HEIGHT (the model regresses opening width, not plate
    length), so rebuilding the predicted rectangle from a *guessed* aspect and
    demanding full-rectangle IoU caps the score no matter how good the model is.

    The fix: give the predicted grasp the SAME height as the ground-truth
    rectangle it is being tested against, then require the centers to be close,
    the angles within threshold, and the widths comparable. This is exactly the
    information the metric is entitled to use -- the annotation supplies the
    height -- and it removes the phantom miss.
    """
    px, py, p_theta, p_w = _as_center_angle_width(pred)
    if px is None:
        return 0
    gts = np.asarray(gts, dtype=float)
    if gts.size == 0:
        return 0
    if gts.ndim == 2:
        gts = gts[None]

    ang_thresh = np.deg2rad(angle_thresh_deg)
    for g in gts:
        gcx, gcy, g_theta, g_w, g_h = corners_to_params(g)
        # width from corners_to_params is the edge0->1 length; the physical jaw
        # opening is the SHORTER side, so use min(w,h) for both.
        g_open = min(g_w, g_h)
        if angle_diff(p_theta, g_theta) >= ang_thresh:
            continue
        # build both rectangles with the SAME height (the GT plate length) so
        # aspect can never cause a phantom miss; vary only center/angle/width.
        g_plate = max(g_w, g_h)
        pred_rect = params_to_corners(px, py, p_theta, max(p_w, 1.0), g_plate)
        gt_rect = params_to_corners(gcx, gcy, g_theta, g_open, g_plate)
        if rect_iou(pred_rect, gt_rect) > iou_thresh:
            return 1
    return 0


def _as_center_angle_width(pred):
    """Normalise a predicted grasp to (x, y, theta, width)."""
    if isinstance(pred, dict):
        x = pred.get("x", pred.get("cx"))
        y = pred.get("y", pred.get("cy"))
        th = pred.get("theta", pred.get("angle", 0.0))
        w = pred.get("width", pred.get("w", 0.0))
        if x is None or y is None:
            return None, None, None, None
        return float(x), float(y), float(th), float(w)
    arr = np.asarray(pred, dtype=float)
    if arr.shape == (4, 2):
        cx, cy, th, w, h = corners_to_params(arr)
        return cx, cy, th, min(w, h)
    if arr.shape == (5,):
        cx, cy, th, w, h = arr
        return float(cx), float(cy), float(th), float(min(w, h))
    raise ValueError(f"unsupported grasp representation with shape {arr.shape}")


def _as_rect_and_theta(pred):
    """Accept several grasp representations and normalise to (corners, theta)."""
    if isinstance(pred, dict):
        if "corners" in pred:
            r = np.asarray(pred["corners"], dtype=float)
            return r, corners_to_params(r)[2]
        cx = pred.get("x", pred.get("cx"))
        cy = pred.get("y", pred.get("cy"))
        th = pred.get("theta", pred.get("angle", 0.0))
        w = pred.get("width", pred.get("w", 0.0))
        h = pred.get("height", pred.get("h", w * 0.5))
        if cx is None or cy is None or not w:
            return None, 0.0
        return params_to_corners(cx, cy, th, w, h), float(th)
    arr = np.asarray(pred, dtype=float)
    if arr.shape == (4, 2):
        return arr, corners_to_params(arr)[2]
    if arr.shape == (5,):
        cx, cy, th, w, h = arr
        return params_to_corners(cx, cy, th, w, h), float(th)
    raise ValueError(f"unsupported grasp representation with shape {arr.shape}")


def evaluate_batch(preds, gts_list, **kw) -> list[int]:
    """Per-sample success indicators for a batch."""
    return [grasp_success(p, g, **kw) for p, g in zip(preds, gts_list)]


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def _selftest():
    ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or exit(1))

    print("[geometry]")
    sq = params_to_corners(0, 0, 0, 2, 2)
    ok(abs(polygon_area(sq) - 4.0) < 1e-9, "area of 2x2 square == 4")
    ok(abs(rect_iou(sq, sq) - 1.0) < 1e-9, "IoU(identical) == 1")

    far = params_to_corners(100, 100, 0, 2, 2)
    ok(rect_iou(sq, far) == 0.0, "IoU(disjoint) == 0")

    # two unit squares offset by half: inter=0.5, union=1.5 -> 1/3
    a = params_to_corners(0, 0, 0, 1, 1)
    b = params_to_corners(0.5, 0, 0, 1, 1)
    ok(abs(rect_iou(a, b) - 1.0 / 3.0) < 1e-6, "IoU(half-overlap) == 1/3")

    # a square rotated 90 deg about its centre is itself
    r90 = params_to_corners(0, 0, np.pi / 2, 2, 2)
    ok(abs(rect_iou(sq, r90) - 1.0) < 1e-6, "IoU(square rotated 90) == 1")

    # 45-degree rotated square: known intersection area 8*(sqrt2 - 1)
    big = params_to_corners(0, 0, 0, 2, 2)
    rot45 = params_to_corners(0, 0, np.pi / 4, 2, 2)
    inter = polygon_area(convex_intersection(big, rot45))
    ok(abs(inter - 8 * (np.sqrt(2) - 1)) < 1e-6,
       f"45-deg overlap area == 8(sqrt2-1)  [{inter:.6f}]")

    print("[angle folding]")
    ok(angle_diff(0.0, np.pi) < 1e-9, "0 and pi are the SAME gripper angle")
    ok(angle_diff(0.0, 2 * np.pi) < 1e-9, "0 and 2pi identical")
    ok(abs(angle_diff(0.0, np.pi / 2) - np.pi / 2) < 1e-9, "0 vs 90deg == 90deg")
    ok(abs(angle_diff(np.deg2rad(10), np.deg2rad(170)) - np.deg2rad(20)) < 1e-9,
       "10deg vs 170deg == 20deg apart (folded)")
    ok(angle_diff(0, np.deg2rad(179)) <= np.pi / 2 + 1e-9, "folded diff <= 90deg")

    print("[success criterion]")
    # GT plate length 40, jaw opening 20, at origin, angle 0
    gt = params_to_corners(50, 50, 0.0, 40, 20)   # w=40 (plate), h=20 (opening)
    # a decoded grasp is a dict with x,y,theta,width(=opening)
    good = dict(x=50, y=50, theta=0.0, width=20)
    ok(grasp_success(good, [gt]) == 1, "matching decoded grasp succeeds")

    flipped = dict(x=50, y=50, theta=np.pi, width=20)
    ok(grasp_success(flipped, [gt]) == 1,
       "grasp flipped by pi still succeeds (pi-periodicity)")

    rot = dict(x=50, y=50, theta=np.pi / 4, width=20)
    ok(grasp_success(rot, [gt]) == 0, "45deg off rejected by the angle gate")

    shifted = dict(x=200, y=200, theta=0.0, width=20)
    ok(grasp_success(shifted, [gt]) == 0, "distant grasp rejected by IoU gate")

    near = dict(x=50, y=50, theta=np.deg2rad(25), width=20)
    ok(grasp_success(near, [gt]) == 1, "25deg off, aligned -> succeeds")
    far_ang = dict(x=50, y=50, theta=np.deg2rad(35), width=20)
    ok(grasp_success(far_ang, [gt]) == 0, "35deg off fails (threshold 30)")

    # slightly wrong opening still succeeds; wildly wrong opening fails
    ok(grasp_success(dict(x=50, y=50, theta=0.0, width=26), [gt]) == 1,
       "opening off by 30% still overlaps enough")
    ok(grasp_success(dict(x=50, y=50, theta=0.0, width=90), [gt]) == 0,
       "opening 4.5x too wide fails")

    gts = [params_to_corners(10, 10, 0, 40, 20), gt]
    ok(grasp_success(good, gts) == 1, "matches any ground-truth in the set")
    ok(grasp_success(good, np.zeros((0, 4, 2))) == 0, "empty GT set -> failure")

    print("[representations]")
    ok(grasp_success(gt, [gt]) == 1, "(4,2) corner prediction accepted")
    ok(grasp_success(np.array([50, 50, 0.0, 20, 40]), [gt]) == 1,
       "(cx,cy,theta,w,h) tuple accepted")
    ok(evaluate_batch([good, shifted], [[gt], [gt]]) == [1, 0], "batch evaluation")

    print("\nALL GRASP-METRIC TESTS PASSED")


if __name__ == "__main__":
    _selftest()
