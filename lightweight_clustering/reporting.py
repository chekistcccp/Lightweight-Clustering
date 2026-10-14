"""Single-file, image-free, locally stored research results.

All reported numbers are computed from evaluation returns, not from raw images.
Never submit the generated experiment_results.json to Git.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import os
import subprocess
from pathlib import Path
import tempfile

import numpy as np

from .metrics_v2 import numeric_fields

METRICS = numeric_fields()
FORBIDDEN_KEYS = frozenset({"image", "images", "mask", "masks", "score_grid",
                            "heatmap", "heatmaps", "features", "raw_volume",
                            "predictions", "voxel_array"})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_json(value, key="root"):
    """Validate JSON strictly; reject image tensors, secrets, NaN/Infinity."""
    if isinstance(value, np.ndarray):
        raise ValueError("Raw arrays cannot be embedded in summary JSON at "+key)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            sk = str(k)
            if sk.lower() in FORBIDDEN_KEYS:
                raise ValueError("Forbidden raw image/prediction field: "+sk)
            out[sk] = clean_json(v, key+"."+sk)
        return out
    if isinstance(value, (list, tuple)):
        return [clean_json(v, key+"[]") for v in value]
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError("Unexpected value in result: "+key+" ("+type(value).__name__+")")


def describe(values):
    xs = np.asarray(values, dtype=np.float64)
    if not len(xs):
        return dict(n=0, mean=None, std=None, median=None, minimum=None, maximum=None)
    return dict(n=len(xs), mean=float(xs.mean()), std=float(xs.std(ddof=1)) if len(xs)>1 else 0.0,
                median=float(np.median(xs)), minimum=float(xs.min()), maximum=float(xs.max()))


def aggregate(methods, rows):
    result = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            raise ValueError("No patient metrics for "+method)
        data = dict(patients=len(selected),
                    evaluated_slices=sum(int(r["slices"]) for r in selected),
                    positive_pixels=sum(int(r["positive_pixels"]) for r in selected),
                    negative_pixels=sum(int(r["negative_pixels"]) for r in selected),
                    patients_without_positive_pixels=sum(int(r["positive_pixels"] == 0) for r in selected),
                    body_valid_pixels=sum(int(r.get("body_valid_pixels",0)) for r in selected),
                    body_excluded_positive_pixels=sum(
                        int(r.get("body_excluded_positive_pixels",0)) for r in selected))
        for metric in METRICS:
            vals = [float(r[metric]) for r in selected if r.get(metric) is not None]
            data[metric] = describe(vals)
        result[method] = data
    return result


def paired_differences(methods, rows, base="patch_knn"):
    per_method = {m: {r["patient"]: r for r in rows if r["method"] == m} for m in methods}
    base_rows = per_method[base]
    output = {}
    for method in methods:
        if method == base:
            continue
        output[method+"_minus_"+base] = {}
        for metric in METRICS:
            diffs = []
            for patient, ref in base_rows.items():
                comp = per_method[method].get(patient)
                if comp is not None and ref.get(metric) is not None and comp.get(metric) is not None:
                    diffs.append(float(comp[metric]) - float(ref[metric]))
            stats = describe(diffs)
            stats["fraction_patients_improved"] = (
                float(np.mean(np.asarray(diffs) > 0)) if diffs else None
            )
            # For false-positive rate a negative delta is favorable; raw delta
            # sign is reported rather than silently reversing direction.
            output[method+"_minus_"+base][metric] = stats
    return output


def ensure_not_git_trackable(path):
    """Inside a Git checkout, refuse to write result files unless ignored.

    Prevents a custom --output path from producing an easily committed report.
    Does not make Git itself a secure data store or disable a user's explicit
    force-add / --no-verify overrides.
    """
    target = Path(path).resolve()
    try:
        run = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=4, check=True)
    except (OSError, subprocess.SubprocessError):
        return
    root = Path(run.stdout.strip()).resolve()
    if not target.is_relative_to(root):
        return  # External directory, outside local Git worktree.
    result = subprocess.run(["git", "check-ignore", "-q", "--", str(target)],
                            cwd=root, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=4)
    if result.returncode != 0:
        raise RuntimeError("Result would be Git-trackable: "+str(target)+
                           ". Use outputs/ (ignored) or a directory outside the repository.")


def export_report(target, *, methods, rows, metadata, diagnostics, timings,
                  splits, started_at, completed_at=None):
    """Atomic single-file export. Does not write any independent metric CSVs."""
    target = Path(target)
    if target.name != "experiment_results.json":
        raise ValueError("Canonical result filename must be experiment_results.json")
    ensure_not_git_trackable(target)
    if not rows:
        raise ValueError("Cannot generate completed report without cases")
    if set(r["method"] for r in rows) != set(methods):
        raise ValueError("Incomplete methods: refuse final result export")
    patients = {m: {r["patient"] for r in rows if r["method"] == m} for m in methods}
    if len({frozenset(v) for v in patients.values()}) != 1:
        raise ValueError("Missing methods for a patient")
    if not set(next(iter(patients.values()))) == set(splits["test"]):
        raise ValueError("Missing/extra test patients in result report")
    payload = dict(
        report_schema_version="2.0.0", status="completed",
        started_at_utc=started_at,
        completed_at_utc=completed_at or utc_timestamp(),
        task="abnormal-only colorectal CT pixel localization",
        contains_images=False, contains_pixel_arrays=False,
        main_result_file="experiment_results.json",
        reporting_note=("All numerical measurements, full/body ROI, top-area budgets, "
                        "paired deltas and sampled numeric PR points in this one local JSON; "
                        "CT, images, pixel heatmaps and feature tensors excluded"),
        metadata=metadata,
        patient_split=dict(seed=splits["seed"], train_count=len(splits["train"]),
                           test_count=len(splits["test"]),
                           train=list(splits["train"]), test=list(splits["test"])),
        cluster_diagnostics=diagnostics,
        timings_seconds=timings,
        metrics=aggregate(methods, rows),
        paired_deltas_vs_patch_knn=paired_differences(methods, rows),
        per_case_metrics=sorted(rows, key=lambda r:(str(r["patient"]), str(r["method"]))),
        metric_definitions=dict(
            pixel_auroc="Per-patient pixel AUROC; null if no positive or no negative pixels",
            pixel_ap="Per-patient pixel average precision; null if one class absent",
            dice_at_top5="Dice using highest-scored 5% pixels per patient, preset threshold",
            iou_at_top5="IoU using highest-scored 5% pixels per patient",
            precision_at_top5="Precision of highest-scored 5% pixels per patient",
            recall_at_top5="Recall of highest-scored 5% pixels per patient",
            fpr_at_top5="False positive rate among negatives at top-5% cutoff",
            aggregate="Patient-level macro unweighted descriptive statistics",
            paired_deltas="Paired per-patient differences; not a hypothesis test",
            top_area_budgets="top0p1=0.1%, top0p5=0.5%, top1=1%, top5=5% of selected ROI pixels",
            body_metrics="Metric names prefixed body_ use HU-derived body pixels as evaluation ROI",
            full_metrics="Unprefixed metrics use entire axial image",
            pr_curve="21 sampled numerical (recall,precision) points per case; NOT plotted images",
            body_mask_area_fraction="Fraction of all evaluated image pixels covered by HU-derived body",
            tumor_coverage_by_body="Fraction of GT positive pixels falling inside HU-derived body; evaluation-only",
            full_score_policy="Body-aware method scores below minimum body score outside the body for full frame comparison"
        ),
        limitations=[
            "The official TORC MATLAB P-code was not ported or proven equivalent; independent approximation",
            "All CT patients are tumor-positive: no patient-level healthy-vs-cancer AUROC",
            "GT masks used only after scoring for evaluation; 2D resampled pixel metrics",
            "Debug subsamples and stride>1 must not be interpreted as full-data results",
            "No image files, raw voxel arrays, heatmaps, or feature tensors in this report",
            "Pixel AP and AUROC should be compared on the same ROI; full/body results are different evaluation populations",
            "Fixed top-area thresholds are exploratory and are not chosen using ground-truth test masks"
        ],
    )
    serializable = clean_json(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    # No report from a previous successful run should masquerade as this run.
    # A new run deletes the stale canonical result before any processing.
    fd, tmp = tempfile.mkstemp(prefix=".incomplete_result_", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target
