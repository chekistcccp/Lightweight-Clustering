"""Numeric-only patient-level pixel metrics for the predefined size budgets.

One stable sort per ROI computes tie-correct AP and ROC AUC, all top-area
operating points, and a small numeric PR curve. No images leave this module.
"""
import numpy as np

BUDGETS = (0.001, 0.005, 0.01, 0.05)
BUDGET_NAMES = ("top0p1", "top0p5", "top1", "top5")
ROI_NAMES = ("full", "body")
SCALAR_FIELDS = ("pixel_auroc", "pixel_ap")
BUDGET_FIELDS = ("dice", "iou", "precision", "recall", "fpr")
COUNT_FIELDS = ("tp", "fp", "fn", "tn")


def numeric_fields():
    names = list(SCALAR_FIELDS)
    names.extend(f"{metric}_at_{suffix}" for suffix in BUDGET_NAMES for metric in BUDGET_FIELDS)
    names.extend("body_" + key for key in list(names))
    return tuple(names)


def rank_metrics(scores, truth, *, curve_points=21):
    """Score only on already-selected ROI. All input score ties are grouped for AUC/AP."""
    score = np.asarray(scores, dtype=np.float32).ravel()
    gt = np.asarray(truth, dtype=bool).ravel()
    if score.size != gt.size or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("Invalid ROI scores/labels (empty, non-finite or shape mismatch)")
    n = score.size
    pos = int(np.count_nonzero(gt)); neg = int(n-pos)
    order = np.argsort(-score, kind="stable")  # deterministic, no GT-derived threshold
    sorted_score = score[order]
    cumulative_tp = np.cumsum(gt[order], dtype=np.int64)
    ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tps = cumulative_tp[ends]
    fps = ends + 1 - tps
    out = dict(pixel_auroc=None, pixel_ap=None, valid_pixels=n,
               positive_pixels=pos, negative_pixels=neg)
    curve = []
    if pos > 0 and neg > 0:
        precision = tps.astype(np.float64) / (ends+1)
        recalls = tps.astype(np.float64) / pos
        delta_pos = np.diff(np.r_[0, tps])
        out["pixel_ap"] = float(np.sum(precision * delta_pos / pos))
        out["pixel_auroc"] = float(np.trapezoid(np.r_[0, tps/pos], np.r_[0, fps/neg]))
        for target in np.linspace(0., 1., num=curve_points):
            if target == 0.:
                curve.append(dict(target_recall=0., recall=0., precision=1.))
                continue
            i = min(int(np.searchsorted(recalls, target, side="left")), len(recalls)-1)
            curve.append(dict(target_recall=float(target), recall=float(recalls[i]),
                              precision=float(precision[i])))
    for budget, suffix in zip(BUDGETS, BUDGET_NAMES):
        count = max(1, int(np.ceil(n * budget)))
        tp = int(cumulative_tp[count-1])
        fp = count-tp
        fn = pos-tp
        tn = neg-fp
        suffix_values = dict(tp=tp,fp=fp,fn=fn,tn=tn,
                             dice=(float(2*tp/(count+pos)) if pos else None),
                             iou=(float(tp/(count+pos-tp)) if pos else None),
                             precision=float(tp/count),
                             recall=(float(tp/pos) if pos else None),
                             fpr=(float(fp/neg) if neg else None))
        for name,value in suffix_values.items():
            out[f"{name}_at_{suffix}"] = value
    return out, curve


def patient_metrics(score, gt, body):
    """Return scalar row + PR curve points. Body ROI is separate from the full frame."""
    score = np.asarray(score, dtype=np.float32)
    gt = np.asarray(gt, dtype=bool)
    body = np.asarray(body, dtype=bool)
    if score.shape != gt.shape or score.shape != body.shape:
        raise ValueError("image/label/ROI shape mismatch")
    if not body.any():
        raise ValueError("Empty body mask; cannot evaluate ROI")
    full,full_curve = rank_metrics(score.ravel(),gt.ravel())
    restricted,body_curve = rank_metrics(score[body],gt[body])
    row = dict(full)
    for key,value in restricted.items():
        row["body_"+key]=value
    row.update(body_mask_area_fraction=float(body.mean()),
               tumor_coverage_by_body=(float(np.count_nonzero(gt&body)/gt.sum())
                                       if gt.any() else None),
               body_excluded_positive_pixels=int(np.count_nonzero(gt&~body)))
    return row, dict(full=full_curve,body=body_curve)
