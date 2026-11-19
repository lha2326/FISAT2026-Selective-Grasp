"""
object_eval.py
==============
Per-object simulation and evaluation for the selective-grasp paper. Everything
here answers the paper's question -- "when NOT to grasp?" -- NOT GR-ConvNet's
question ("is the grasp accurate?"). The distinction drives every function:

  * GR-ConvNet reports success rate. We report, per object, whether the
    abstention rule EXECUTES or REJECTS, and whether that decision was right.
  * GR-ConvNet's physical trial asks "did the grasp hold?". Our lightweight
    trial asks "if we obey the rule, is the failure rate among EXECUTED grasps
    within budget?" -- an empirical check of the LTT guarantee, per object.

Three deliverables, matching the three requests:

  1. per_object_visual_records()  -> data for image + predicted grasp +
     accept/reject label + uncertainty, one record per object.
  2. simulate_execution()          -> a physics-free "execution" trial on the
     Jacquard test set: apply tau, measure realised risk on executed grasps,
     confirm it sits under alpha. This is the LTT guarantee made concrete.
  3. per_object_table()            -> success / rejection / executed-risk broken
     down by object, exposing which objects the rule protects against.

No physics engine. The Jacquard success label already encodes whether a
predicted grasp matches the annotated grasp set under the rectangle metric, so
"executing" a grasp on the test set means reading that label -- deterministic,
reproducible, and free of a PyBullet dependency that would not run cleanly on
Colab. Where a physical simulator would add value (contact dynamics, slip) is
noted in the docstrings so the paper can scope it as future work honestly.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# =========================================================================== #
# 1. Per-object visual records
# =========================================================================== #
def per_object_visual_records(idx, obj_id, success, score, tau,
                              max_per_object: int = 1, bias="rejected"):
    """One record per object for the qualitative figure.

    Returns a list of dicts with the fields a plotting routine needs:
        obj_id, idx (dataset index, to fetch the image), executed (bool),
        correct (bool), score, and outcome in {accepted_correct,
        accepted_wrong, rejected_would_fail, rejected_would_succeed}.

    The four outcomes are the whole point. GR-ConvNet figures show correct vs
    incorrect grasps; ours must show the DECISION crossed with the truth,
    because "rejected a grasp that would have failed" is the behaviour the paper
    claims and "accepted one that failed" is the residual risk it must own.

    `bias` picks which grasp to show per object when it has both executed and
    rejected grasps: "rejected" (default) showcases a rejection; "executed"
    showcases an accepted grasp; "none" takes the first. For a figure that mixes
    green and red borders, generate records with several biases (or use
    figure_per_object's ensure_both, which pulls executed and rejected records
    from a bias="none" set).
    """
    idx = np.asarray(idx)
    obj_id = np.asarray(obj_id)
    success = np.asarray(success).astype(int)
    score = np.asarray(score)

    executed = score <= tau
    records = []
    for oid in np.unique(obj_id):
        mask = obj_id == oid
        # per-object statistics, attached to every record for this object so a
        # panel can show "this object's grasps succeed X% of the time" and
        # "the rule executes Y% of them"
        obj_success_rate = float(success[mask].mean())
        obj_coverage = float(executed[mask].mean())
        n_obj = int(mask.sum())
        # pick which grasp of this object to display
        if bias == "rejected":
            order = np.argsort(~(~executed[mask]))   # rejected first
        elif bias == "executed":
            order = np.argsort(~executed[mask])       # executed first
        else:
            order = np.arange(int(mask.sum()))        # as-is
        chosen = np.where(mask)[0][order][:max_per_object]
        for j in chosen:
            ex = bool(executed[j])
            ok = bool(success[j])
            if ex and ok:
                outcome = "accepted_correct"
            elif ex and not ok:
                outcome = "accepted_wrong"          # residual risk
            elif not ex and not ok:
                outcome = "rejected_would_fail"      # correct rejection
            else:
                outcome = "rejected_would_succeed"   # cost of caution
            records.append(dict(obj_id=int(oid), idx=int(idx[j]),
                                executed=ex, correct=ok,
                                score=float(score[j]), outcome=outcome,
                                obj_success_rate=obj_success_rate,
                                obj_coverage=obj_coverage, n_obj=n_obj))
    return records


def outcome_counts(records):
    """Tally the four decision outcomes across a set of records."""
    counts = {"accepted_correct": 0, "accepted_wrong": 0,
              "rejected_would_fail": 0, "rejected_would_succeed": 0}
    for r in records:
        counts[r["outcome"]] += 1
    return counts


# =========================================================================== #
# 2. Physics-free execution simulation (the LTT guarantee, made concrete)
# =========================================================================== #
def simulate_execution(success, score, tau, alpha=None):
    """'Execute' every grasp the rule accepts; measure realised risk.

    On the Jacquard test set a grasp's success label already says whether the
    predicted grasp matches the annotation under the rectangle metric, so
    executing the accepted subset means reading those labels. Returns:

        n_total, n_executed, coverage,
        executed_risk   -- failure rate AMONG executed grasps (the quantity the
                           budget constrains),
        rejected_risk   -- failure rate among rejected grasps (should be HIGHER;
                           if it isn't, the score isn't ordering by risk),
        within_budget   -- executed_risk <= alpha (None if alpha not given).

    A physical simulator would additionally model contact and slip, so the true
    execution risk could exceed this metric-based estimate; that gap is the
    honest scope of "future work: closed-loop physical trials".
    """
    success = np.asarray(success).astype(int)
    fail = 1 - success
    score = np.asarray(score)

    executed = score <= tau
    n_exec = int(executed.sum())
    n_rej = int((~executed).sum())

    exec_risk = float(fail[executed].mean()) if n_exec else 0.0
    rej_risk = float(fail[~executed].mean()) if n_rej else 0.0

    out = dict(
        n_total=len(success),
        n_executed=n_exec,
        n_rejected=n_rej,
        coverage=n_exec / max(len(success), 1),
        executed_risk=exec_risk,
        rejected_risk=rej_risk,
        risk_reduction=float(fail.mean()) - exec_risk,   # vs executing all
    )
    if alpha is not None:
        out["alpha"] = alpha
        out["within_budget"] = exec_risk <= alpha + 1e-9
    return out


def simulate_execution_curve(success, score, taus):
    """Run simulate_execution across a sweep of thresholds, for a plot."""
    return [dict(tau=float(t), **simulate_execution(success, score, t))
            for t in taus]


# =========================================================================== #
# 3. Per-object breakdown
# =========================================================================== #
def per_object_table(obj_id, success, score, tau, alpha=None,
                     min_samples: int = 1, sort_by="executed_risk"):
    """Success / rejection / executed-risk per object.

    Exposes which objects the detector struggles with and whether the rule
    rejects them. An object with low raw success but high rejection rate is a
    WIN for the paper: the rule recognised a hard object and abstained. An
    object with low success AND low rejection is the residual-risk case worth
    showing.
    """
    obj_id = np.asarray(obj_id)
    success = np.asarray(success).astype(int)
    fail = 1 - success
    score = np.asarray(score)
    executed = score <= tau

    rows = []
    for oid in np.unique(obj_id):
        m = obj_id == oid
        n = int(m.sum())
        if n < min_samples:
            continue
        n_exec = int(executed[m].sum())
        exec_fail = fail[m & executed]
        rows.append({
            "obj_id": int(oid),
            "n": n,
            "raw_success": float(success[m].mean()),
            "rejection_rate": float((~executed[m]).mean()),
            "coverage": n_exec / max(n, 1),
            "executed_risk": float(exec_fail.mean()) if len(exec_fail) else 0.0,
        })
    reverse = sort_by in ("executed_risk", "rejection_rate")
    rows.sort(key=lambda r: r.get(sort_by, 0.0), reverse=reverse)
    return rows


def object_summary(rows, alpha=None):
    """Aggregate the per-object table into a few headline numbers."""
    if not rows:
        return {}
    ex_risks = np.array([r["executed_risk"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    summary = dict(
        n_objects=len(rows),
        mean_coverage=float(cov.mean()),
        mean_executed_risk=float(ex_risks.mean()),
        worst_executed_risk=float(ex_risks.max()),
        objects_fully_rejected=int((cov == 0).sum()),
    )
    if alpha is not None:
        summary["objects_over_budget"] = int((ex_risks > alpha + 1e-9).sum())
        summary["frac_objects_within_budget"] = \
            float((ex_risks <= alpha + 1e-9).mean())
    return summary


def alpha_sweep(score, loss, obj_id, alphas, delta=0.10,
                ltt_calibrate=None, simulate=None, per_object=None):
    """Compare candidate risk budgets side by side so alpha can be chosen with
    evidence, not by habit.

    For each alpha it calibrates a threshold by LTT, simulates execution, and
    counts how many objects the rule actually discriminates on (partial
    coverage: neither fully executed nor fully rejected). The columns that
    matter when choosing:

        ltt_valid            - was a threshold certifiable at this alpha and
                               this calibration size? FALSE means the budget is
                               too strict for the data you have.
        coverage             - fraction of grasps executed.
        executed_risk        - realised failure rate among executed grasps.
        objs_discriminating  - objects where the rule made a non-trivial choice.
        trivial              - base failure rate <= alpha, so "execute all"
                               already meets the budget and the rule is idle.

    The sweet spot is the strictest alpha that is still ltt_valid AND not
    trivial: there the guarantee holds and the rejection rule is doing real
    work. Pass the selective_grasp functions in (kept as arguments so this
    module has no hard dependency on them).
    """
    import importlib
    if ltt_calibrate is None or simulate is None or per_object is None:
        sg = importlib.import_module("selective_grasp")
        ltt_calibrate = ltt_calibrate or sg.ltt_calibrate_single
        simulate = simulate or simulate_execution
        per_object = per_object or per_object_table

    loss = np.asarray(loss).astype(float)
    base = float(loss.mean())
    rows = []
    for a in alphas:
        res = ltt_calibrate(score, loss, a, delta)
        tau = res.tau if res.valid else float("-inf")
        sim = simulate(1 - loss, score, tau, alpha=a)
        tbl = per_object(obj_id, 1 - loss, score, tau, alpha=a)
        discriminating = sum(1 for r in tbl if 0.0 < r["coverage"] < 1.0)
        rows.append(dict(
            alpha=a,
            ltt_valid=bool(res.valid),
            coverage=round(sim["coverage"], 3),
            executed_risk=round(sim["executed_risk"], 4),
            within_budget=sim.get("within_budget"),
            objs_discriminating=f"{discriminating}/{len(tbl)}",
            trivial=(base <= a + 1e-9),
        ))
    return rows


def grid_search(scores, loss, obj_id, alphas, delta=0.10,
                ltt_calibrate=None):
    """Full (score x alpha) grid for choosing the operating point.

    For every uncertainty score and every candidate budget, report whether LTT
    certifies a threshold, the resulting coverage and executed risk, whether the
    budget is met, and whether it is trivial (base failure rate <= alpha). This
    is the table to scan when picking WHICH score to threshold and at WHAT
    budget -- the two decisions the paper's operating point rests on.

    Unlike alpha_sweep (one score, many alphas), this crosses every score with
    every alpha, so the winning combination is visible in one place.
    """
    import importlib
    if ltt_calibrate is None:
        ltt_calibrate = importlib.import_module("selective_grasp").ltt_calibrate_single

    loss = np.asarray(loss).astype(float)
    base = float(loss.mean())
    grid = []
    for name, sc in scores.items():
        sc = np.asarray(sc)
        for a in alphas:
            res = ltt_calibrate(sc, loss, a, delta)
            tau = res.tau if res.valid else float("-inf")
            sim = simulate_execution(1 - loss, sc, tau, alpha=a)
            grid.append(dict(
                score=name, alpha=a, ltt_valid=bool(res.valid),
                coverage=round(sim["coverage"], 3),
                exec_risk=round(sim["executed_risk"], 4),
                within_budget=sim.get("within_budget"),
                trivial=(base <= a + 1e-9),
            ))
    return grid


def best_operating_point(grid):
    """Highest-coverage (score, alpha) that is certified, within budget, and
    non-trivial. Returns the row, or None with a reason if none qualifies.

    The ranking is deliberate: among rows where the guarantee actually holds and
    the rule is doing real work, more coverage is strictly better -- it means
    executing more grasps at the same risk ceiling. Trivial rows are excluded
    because their coverage is high for the wrong reason (the budget was slack).
    """
    usable = [r for r in grid
              if r["ltt_valid"] and r["within_budget"] and not r["trivial"]]
    if usable:
        best = max(usable, key=lambda r: r["coverage"])
        return best, "highest coverage among certified, within-budget, non-trivial"
    certified = [r for r in grid if r["ltt_valid"] and not r["trivial"]]
    if certified:
        best = max(certified, key=lambda r: r["coverage"])
        return best, ("no row meets budget exactly; showing best certified "
                      "non-trivial (report the small overshoot honestly)")
    return None, ("nothing certifies non-trivially at this calibration size; "
                  "more calibration data is needed -- this is what Jacquard "
                  "buys over Cornell")


def recommend_alpha(sweep_rows):
    """Pick the strictest alpha that is both certifiable and non-trivial.

    Returns (alpha, reason). If none qualifies, falls back to the strictest
    certifiable alpha and says so, or to the loosest alpha if nothing certifies.
    """
    usable = [r for r in sweep_rows if r["ltt_valid"] and not r["trivial"]]
    if usable:
        best = min(usable, key=lambda r: r["alpha"])
        return best["alpha"], "strictest budget that is certifiable and non-trivial"
    certifiable = [r for r in sweep_rows if r["ltt_valid"]]
    if certifiable:
        best = min(certifiable, key=lambda r: r["alpha"])
        return best["alpha"], ("no non-trivial budget certifies at this "
                               "calibration size; using strictest certifiable")
    loosest = max(sweep_rows, key=lambda r: r["alpha"])
    return loosest["alpha"], ("NOTHING certifies at this calibration size; "
                              "report this as a limitation and use more data")


def regime_check(success, alpha) -> dict:
    """Flag whether the detector leaves enough failures for the study to be
    informative at this budget.

    When the base failure rate is at or below alpha, the trivial rule "execute
    everything" already satisfies the budget, so the rejection rule has nothing
    to prove and every per-object executed-risk collapses to ~0. This is not a
    bug; it is a statement about the operating point. On Jacquard, where the
    detector reaches ~93%, the interesting budgets are the STRICT ones (well
    below the ~7% base failure rate). The function returns the base rate, whether
    the budget is non-trivial, and a suggested alpha if it is not.
    """
    success = np.asarray(success).astype(int)
    base_fail = float((1 - success).mean())
    non_trivial = base_fail > alpha + 1e-9
    out = dict(base_failure_rate=base_fail, alpha=alpha,
               budget_non_trivial=non_trivial)
    if not non_trivial:
        # suggest a budget that actually forces the rule to discriminate:
        # roughly half the base failure rate, so "execute all" is NOT a solution
        out["suggested_alpha"] = round(max(base_fail / 2.0, 0.005), 3)
        out["note"] = (
            "Base failure rate <= alpha: 'execute all' already meets the "
            "budget, so the rejection rule is not exercised and per-object "
            "risks read ~0. Use a stricter alpha (suggested) to make the "
            "selective-prediction contribution visible.")
    return out


def rejection_breakdown(success, score, tau):
    """Diagnose over- vs under-rejection with one call.

    Splits the rejected pile into grasps that WOULD have succeeded (the cost of
    caution -- over-rejection) and grasps that would have failed (correct
    rejections), and the executed pile into correct and wrong. Returns counts,
    the four rates, and two headline numbers:

        over_rejection_rate  -- of everything rejected, the fraction that would
                                have succeeded. High means the threshold is too
                                strict: it is throwing away good grasps.
        precision_of_rejection -- of everything rejected, the fraction that
                                would indeed have failed. This is the number to
                                report: "when the rule says no, it is right X%
                                of the time."
    """
    success = np.asarray(success).astype(int)
    fail = 1 - success
    executed = np.asarray(score) <= tau
    rejected = ~executed

    n_rej = int(rejected.sum())
    n_exec = int(executed.sum())
    rej_would_fail = int(fail[rejected].sum())
    rej_would_succeed = int(success[rejected].sum())
    exec_correct = int(success[executed].sum())
    exec_wrong = int(fail[executed].sum())

    return dict(
        n_total=len(success),
        n_executed=n_exec,
        n_rejected=n_rej,
        rejected_would_fail=rej_would_fail,
        rejected_would_succeed=rej_would_succeed,
        executed_correct=exec_correct,
        executed_wrong=exec_wrong,
        over_rejection_rate=(rej_would_succeed / n_rej) if n_rej else 0.0,
        precision_of_rejection=(rej_would_fail / n_rej) if n_rej else 0.0,
        coverage=n_exec / max(len(success), 1),
    )


def rejection_summary_line(success, score, tau) -> str:
    """A one-line human-readable summary to print above the figure."""
    b = rejection_breakdown(success, score, tau)
    if b["n_rejected"] == 0:
        return "Nothing rejected at this threshold."
    return (
        f"Rejected {b['n_rejected']}/{b['n_total']} grasps: "
        f"{b['precision_of_rejection']*100:.0f}% would truly have failed "
        f"(correct rejections), "
        f"{b['over_rejection_rate']*100:.0f}% would have succeeded "
        f"(over-rejection, the cost of caution). "
        f"Executed {b['n_executed']}: "
        f"{b['executed_correct']} correct, {b['executed_wrong']} wrong."
    )


# =========================================================================== #
# 4. Rendering the per-object figure (the piece that draws actual images)
# =========================================================================== #
def render_record(dataset, model, idx, device="cpu", width_scale=100.0):
    """Fetch image + predicted grasp + annotations for ONE dataset index.

    Returns the dict make_figures.draw_panel expects: image, quality, pred, gts.
    This is the bridge the earlier notebook left as a sketch -- it actually runs
    the detector on the stored image so the panel shows a real prediction, not a
    placeholder.
    """
    import torch
    from models_grconvnet import post_process
    from train import decode_grasp
    from metrics_grasp import corners_to_params

    sample = dataset[idx]
    x = sample["image"]
    x = x[None] if hasattr(x, "ndim") else x
    if not hasattr(x, "to"):
        x = torch.as_tensor(np.asarray(x))[None]
    with torch.no_grad():
        out = model(x.to(device))
    if isinstance(out, dict):                       # evidential wrapper
        pos, cos, sin, wd = out["pos"], out["cos"], out["sin"], out["width"]
    else:
        pos, cos, sin, wd = out
    q, ang, w = post_process(pos, cos, sin, wd)
    g = decode_grasp(q, ang, w)

    # RGB for display: first three channels, H,W,C
    img = np.asarray(sample["image"])
    if img.shape[0] in (3, 4):
        img = np.transpose(img[:3], (1, 2, 0))
    img = np.clip(img, 0, 1)

    gts = []
    for rect in np.asarray(sample["gt_grasps"]):
        cx, cy, th, ww, hh = corners_to_params(rect)
        gts.append(dict(x=float(cx), y=float(cy), theta=float(th),
                        width=float(min(ww, hh))))
    return dict(image=img, quality=q, pred=g, gts=gts, alts=[])


def figure_per_object(dataset, model, records, out_path, device="cpu",
                      n_cols=4, max_panels=12, prefer=("rejected_would_fail",
                                                       "accepted_wrong"),
                      balanced=True, ensure_both=True):
    """Grid of per-object panels showing a VARIETY of decision outcomes.

    `records` come from per_object_visual_records(). Two knobs control the mix:

    `balanced=True` draws round-robin across the four outcomes (accepted-correct,
    rejected-would-fail, accepted-wrong, rejected-would-succeed) so the grid
    isn't dominated by whichever outcome is most common.

    `ensure_both=True` guarantees the grid contains BOTH executed (green border)
    and rejected (red border) panels when each exists in the data. At a strict
    threshold almost every OBJECT is rejected, so even balanced sampling can come
    out all-red; this reserves half the panels for executed grasps if any exist,
    so the figure always illustrates both colours of decision. If one class is
    absent entirely (e.g. nothing was executed at this threshold), it simply
    fills from the other -- and that all-red result is itself the honest message
    that the threshold is very strict.

    A green border means executed, red means rejected; the caption states the
    outcome. Falls back to the synthetic renderer only if the dataset/model
    can't produce an image, so the layout is always viewable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import draw_panel

    def _round_robin(recs, order, k):
        buckets = {o: [r for r in recs if r["outcome"] == o] for o in order}
        present = [o for o in order if buckets[o]]
        out, i = [], 0
        while len(out) < k and any(buckets[o] for o in present):
            o = present[i % len(present)]
            if buckets[o]:
                out.append(buckets[o].pop(0))
            i += 1
        return out

    order = ["accepted_correct", "rejected_would_fail",
             "accepted_wrong", "rejected_would_succeed"]

    if ensure_both:
        # reserve up to half the grid for each decision, so both borders show
        executed = [r for r in records if r["executed"]]
        rejected = [r for r in records if not r["executed"]]
        half = max_panels // 2
        take_exec = _round_robin(executed, order, min(half, len(executed)))
        take_rej = _round_robin(rejected, order, max_panels - len(take_exec))
        # if executed is short, backfill from rejected (and vice versa)
        chosen = take_exec + take_rej
        if len(chosen) < max_panels:
            leftover = [r for r in (executed + rejected) if r not in chosen]
            chosen += leftover[:max_panels - len(chosen)]
        chosen = chosen[:max_panels]
    elif balanced:
        chosen = _round_robin(records, order, max_panels)
    else:
        pref = [r for r in records if r["outcome"] in prefer]
        rest = [r for r in records if r["outcome"] not in prefer]
        chosen = (pref + rest)[:max_panels]
    if not chosen:
        raise ValueError("no records to render")

    from collections import Counter
    spread = Counter(r["outcome"] for r in chosen)
    n_exec = sum(1 for r in chosen if r["executed"])
    print(f"figure: {n_exec} executed (green), {len(chosen)-n_exec} rejected (red)")
    print("outcome mix:", {k: spread[k] for k in order if spread.get(k)})

    n = len(chosen)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    label = {"rejected_would_fail": "Rejected — would fail  (correct)",
             "accepted_wrong": "Accepted — WRONG  (residual risk)",
             "accepted_correct": "Accepted — correct",
             "rejected_would_succeed": "Rejected — would succeed  (cost)"}

    for ax, r in zip(axes, chosen):
        try:
            s = render_record(dataset, model, r["idx"], device=device)
        except Exception:
            from make_figures import render_sample_demo
            s = render_sample_demo(r["idx"])
        # two-line caption: the decision outcome, then this object's grasp
        # success rate and how much of it the rule executes. The success rate is
        # the number requested; showing it next to the decision makes the
        # over/under-rejection visible at a glance.
        sr = r.get("obj_success_rate")
        cov = r.get("obj_coverage")
        stat = ""
        if sr is not None:
            stat = f"\nsuccess {sr*100:.0f}%"
            if cov is not None:
                stat += f" · executed {cov*100:.0f}%"
        draw_panel(ax, s, accepted=r["executed"], success=r["correct"],
                   subtitle=f"obj {r['obj_id']} · {label[r['outcome']]}{stat}")
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cross_dataset_comparison(datasets, alphas, delta=0.10, ltt_calibrate=None):
    """Compare over-rejection and coverage across datasets at several budgets.

    `datasets` maps a name to (success_array, best_score_array). For each dataset
    and alpha it reports coverage, executed risk, and over-rejection, so the
    paper can show WHERE each dataset's operating regime is: a weaker detector
    (Cornell) does real work at moderate alpha, a stronger one (Jacquard) needs
    a strict alpha before the rejection rule is exercised at all.

    This is the table behind the "two datasets, two complementary regimes"
    argument -- the strongest framing available for using both.
    """
    import importlib
    if ltt_calibrate is None:
        ltt_calibrate = importlib.import_module("selective_grasp").ltt_calibrate_single

    rows = []
    for name, (success, score) in datasets.items():
        success = np.asarray(success).astype(int)
        loss = 1 - success
        base = float(success.mean())
        for a in alphas:
            res = ltt_calibrate(np.asarray(score), loss, a, delta)
            tau = res.tau if res.valid else float("-inf")
            b = rejection_breakdown(success, score, tau)
            rows.append(dict(
                dataset=name, base_success=round(base, 3), alpha=a,
                ltt_valid=bool(res.valid),
                coverage=round(b["coverage"], 3),
                over_rejection=round(b["over_rejection_rate"], 3),
                rej_precision=round(b["precision_of_rejection"], 3),
                trivial=((1 - base) <= a + 1e-9),
            ))
    return rows


# =========================================================================== #
# Self-test
# =========================================================================== #
def _selftest():
    def ok(c, m):
        print(f"  {'PASS' if c else 'FAIL'}  {m}")
        if not c:
            raise SystemExit(1)

    rng = np.random.default_rng(0)
    n = 400
    # scores correlate with failure: higher score -> more likely to fail
    u = rng.random(n)
    fail = (rng.random(n) < u).astype(int)
    success = 1 - fail
    score = u + 0.15 * rng.normal(size=n)
    obj_id = rng.integers(0, 20, n)          # 20 objects

    print("[execution simulation]")
    tau = np.quantile(score, 0.6)            # accept the 60% lowest-risk
    sim = simulate_execution(success, score, tau, alpha=0.15)
    ok(0 < sim["coverage"] < 1, "coverage strictly between 0 and 1")
    ok(sim["executed_risk"] < sim["rejected_risk"],
       "executed risk < rejected risk (score orders by risk)")
    ok(sim["risk_reduction"] > 0, "abstention reduces risk vs executing all")
    ok(sim["n_executed"] + sim["n_rejected"] == sim["n_total"],
       "executed + rejected == total")

    print("[monotone coverage sweep]")
    taus = np.quantile(score, np.linspace(0.1, 0.9, 9))
    curve = simulate_execution_curve(success, score, taus)
    covs = [c["coverage"] for c in curve]
    ok(all(covs[i] <= covs[i+1] + 1e-9 for i in range(len(covs)-1)),
       "coverage increases as tau rises")
    risks = [c["executed_risk"] for c in curve]
    ok(risks[0] <= risks[-1] + 1e-9,
       "executed risk grows as more (riskier) grasps are admitted")

    print("[per-object visual records]")
    recs = per_object_visual_records(np.arange(n), obj_id, success, score, tau)
    ok(len(recs) == len(np.unique(obj_id)), "one record per object by default")
    outs = {r["outcome"] for r in recs}
    ok(outs <= {"accepted_correct", "accepted_wrong",
                "rejected_would_fail", "rejected_would_succeed"},
       "every record has a valid outcome label")
    counts = outcome_counts(recs)
    ok(sum(counts.values()) == len(recs), "outcome tally covers all records")

    print("[per-object table]")
    tbl = per_object_table(obj_id, success, score, tau, alpha=0.15)
    ok(len(tbl) == len(np.unique(obj_id)), "one row per object")
    for r in tbl:
        ok(0 <= r["executed_risk"] <= 1, f"obj {r['obj_id']} risk in [0,1]")
        break
    ok(all(tbl[i]["executed_risk"] >= tbl[i+1]["executed_risk"]
           for i in range(len(tbl)-1)), "table sorted by executed risk desc")

    summ = object_summary(tbl, alpha=0.15)
    ok(summ["n_objects"] == len(tbl), "summary counts all objects")
    ok(0 <= summ["frac_objects_within_budget"] <= 1,
       "fraction within budget in [0,1]")

    print("[edge cases]")
    # tau below every score -> reject everything
    s0 = simulate_execution(success, score, score.min() - 1)
    ok(s0["n_executed"] == 0 and s0["coverage"] == 0.0, "reject-all handled")
    # tau above every score -> execute everything, risk == base rate
    s1 = simulate_execution(success, score, score.max() + 1)
    ok(s1["coverage"] == 1.0 and abs(s1["executed_risk"] - fail.mean()) < 1e-9,
       "execute-all risk equals base failure rate")

    print("[regime check]")
    all_good = np.ones(100, int)
    mostly_good = np.concatenate([np.ones(93, int), np.zeros(7, int)])
    r1 = regime_check(mostly_good, alpha=0.10)
    ok(not r1["budget_non_trivial"], "7% fail vs alpha=0.10 -> trivial budget flagged")
    ok("suggested_alpha" in r1 and r1["suggested_alpha"] < 0.07,
       "a stricter alpha is suggested")
    r2 = regime_check(mostly_good, alpha=0.03)
    ok(r2["budget_non_trivial"], "7% fail vs alpha=0.03 -> non-trivial, no flag")

    print("[alpha sweep]")
    sweep = alpha_sweep(score, fail.astype(float), obj_id,
                        [0.20, 0.15, 0.10, 0.05])
    ok(len(sweep) == 4, "one row per candidate alpha")
    ok(all("ltt_valid" in r and "coverage" in r for r in sweep),
       "sweep rows carry the decision columns")
    # coverage must be monotone non-increasing as alpha tightens
    covs = [r["coverage"] for r in sweep]
    ok(all(covs[i] >= covs[i+1] - 1e-9 for i in range(len(covs)-1)),
       "coverage does not increase as alpha tightens")
    a_rec, reason = recommend_alpha(sweep)
    ok(a_rec in [r["alpha"] for r in sweep], "recommended alpha is one of the candidates")
    ok(isinstance(reason, str) and len(reason) > 0, "recommendation gives a reason")

    print("[grid search]")
    sc_multi = {"a": score, "b": score + 0.1 * rng.normal(size=n),
                "c": score - 0.05 * rng.normal(size=n)}
    grid = grid_search(sc_multi, fail.astype(float), obj_id, [0.20, 0.15, 0.10])
    ok(len(grid) == 3 * 3, "grid has one row per (score, alpha) pair")
    ok(all(set(r) >= {"score","alpha","ltt_valid","coverage","exec_risk",
                      "within_budget","trivial"} for r in grid),
       "each grid row carries the decision columns")
    bop, why = best_operating_point(grid)
    ok(bop is None or (bop["ltt_valid"] and not bop["trivial"]),
       "best operating point is certified and non-trivial (or None)")
    ok(isinstance(why, str) and len(why) > 0, "best point gives a reason")
    # coverage monotone in alpha within a fixed score
    for name in sc_multi:
        rows = [r for r in grid if r["score"] == name]
        rows.sort(key=lambda r: -r["alpha"])
        covs = [r["coverage"] for r in rows]
        ok(all(covs[i] >= covs[i+1] - 1e-9 for i in range(len(covs)-1)),
           f"coverage non-increasing as alpha tightens for score {name}")

    print("[rejection breakdown]")
    b = rejection_breakdown(success, score, tau)
    ok(b["n_executed"] + b["n_rejected"] == b["n_total"], "counts partition")
    ok(b["rejected_would_fail"] + b["rejected_would_succeed"] == b["n_rejected"],
       "rejected pile splits into fail/succeed")
    ok(abs(b["over_rejection_rate"] + b["precision_of_rejection"] - 1.0) < 1e-9,
       "over-rejection and rejection-precision sum to 1")
    ok(0 <= b["over_rejection_rate"] <= 1, "over-rejection rate in [0,1]")
    line = rejection_summary_line(success, score, tau)
    ok(isinstance(line, str) and "Rejected" in line, "summary line renders")
    # a very strict threshold over-rejects; a very loose one does not
    strict = rejection_breakdown(success, score, np.quantile(score, 0.05))
    loose = rejection_breakdown(success, score, np.quantile(score, 0.95))
    ok(strict["over_rejection_rate"] >= loose["over_rejection_rate"] - 1e-9,
       "stricter threshold over-rejects at least as much")

    print("[cross-dataset comparison]")
    s_weak = success
    sc_weak = score
    s_strong = np.concatenate([np.ones(360, int), np.zeros(40, int)])[:n]
    sc_strong = rng.random(n)
    cross = cross_dataset_comparison(
        {"weak": (s_weak, sc_weak), "strong": (s_strong, sc_strong)},
        [0.15, 0.10, 0.05])
    ok(len(cross) == 2 * 3, "one row per (dataset, alpha)")
    ok(all("over_rejection" in r and "coverage" in r for r in cross),
       "cross rows carry over-rejection and coverage")
    ok(all(0 <= r["over_rejection"] <= 1 for r in cross),
       "over-rejection in [0,1] for every row")

    print("\nALL OBJECT-EVAL TESTS PASSED")


if __name__ == "__main__":
    _selftest()
