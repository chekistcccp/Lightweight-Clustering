"""Atomic single-file V3 results with factorial comparisons and sampling-seed stability.

Report is created only after all seed runs finish. No arrays, imaging data,
weights, heatmaps, prediction grids or user results are sent to GitHub.
"""
import json
import os
from pathlib import Path
import tempfile

from .reporting import (aggregate,clean_json,describe,ensure_not_git_trackable,
                        utc_timestamp,METRICS)

FACTORIAL_CONTRASTS=(
    ("body_patch_knn","patch_knn"),
    ("body_diverse_512","body_patch_knn"),
    ("body_random_8192","body_patch_knn"),
    ("diverse_patch_knn","body_random_8192"),
    ("diverse_patch_knn","body_diverse_512"),
    ("torc_approx","diverse_patch_knn"),
    ("local_consensus_knn","diverse_patch_knn"),
    ("torc_consensus","torc_approx"),
    ("torc_consensus","local_consensus_knn"),
)


def compare_pairs(rows, variants=FACTORIAL_CONTRASTS):
    by_method={}
    for row in rows:
        key=(row["method"],row["patient"])
        if key in by_method:
            raise ValueError("Duplicated method/patient in seed")
        by_method[key]=row
    result={}
    for treatment,control in variants:
        patients={name for method,name in by_method if method==treatment}
        patients_ref={name for method,name in by_method if method==control}
        if patients != patients_ref or not patients:
            raise ValueError("Contrast has incomplete patient pairs: "+treatment+"/"+control)
        values={}
        for metric in METRICS:
            diffs=[float(by_method[(treatment,p)][metric])-
                   float(by_method[(control,p)][metric])
                   for p in sorted(patients)
                   if by_method[(treatment,p)].get(metric) is not None and
                      by_method[(control,p)].get(metric) is not None]
            stat=describe(diffs)
            stat["fraction_positive"]=(
                float(sum(v>0 for v in diffs)/len(diffs)) if diffs else None)
            values[metric]=stat
        result[f"{treatment}_minus_{control}"]=values
    return result



def factorial_effects(rows):
    """Per-patient 2x2 contrasts: size, diversity and difference-in-differences.

    Values use the same 4 methods at the same bank-sampling seed.
    This is an exploratory balanced contrast, NOT an independent validation.
    """
    by={(r["method"],r["patient"]):r for r in rows}
    patients=sorted({p for m,p in by if m=="body_patch_knn"})
    patterns=dict(
        bank_size_main=("body_random_8192","body_patch_knn",
                        "diverse_patch_knn","body_diverse_512"),
        diversity_main=("body_diverse_512","body_patch_knn",
                        "diverse_patch_knn","body_random_8192"),
        size_x_diversity_interaction=("diverse_patch_knn","body_random_8192",
                                      "body_diverse_512","body_patch_knn"),
    )
    output={}
    for label,methods4 in patterns.items():
        output[label]={}
        for metric in METRICS:
            values=[]
            for patient in patients:
                q=[by[(method,patient)].get(metric) for method in methods4]
                if any(v is None for v in q):
                    continue
                if label=="size_x_diversity_interaction":
                    delta=(float(q[0])-float(q[1]))-(float(q[2])-float(q[3]))
                else:
                    delta=0.5*((float(q[0])-float(q[1]))+
                               (float(q[2])-float(q[3])))
                values.append(delta)
            output[label][metric]=describe(values)
    return output


def write_multiseed_report(target, *, methods, records, split, metadata,
                           common_timings, started_at):
    """Validate every seed and write one JSON atomically (no partial report)."""
    path=Path(target)
    if path.name!="experiment_results.json":
        raise ValueError("Unexpected canonical result filename")
    ensure_not_git_trackable(path)
    if not records:
        raise ValueError("No finished seeds")
    expected=set(split["test"])
    all_seeds=[]
    prepared=[]
    for record in records:
        seed=int(record["sampling_seed"])
        if seed in all_seeds:
            raise ValueError("Duplicate sampling seed")
        rows=record["rows"]
        observed={(r["method"],r["patient"]) for r in rows}
        required={(m,p) for m in methods for p in expected}
        if observed!=required or len(rows)!=len(required):
            raise ValueError("Incomplete or duplicated patient/method at seed "+str(seed))
        summary=aggregate(methods,rows)
        deltas=compare_pairs(rows)
        factorial=factorial_effects(rows)
        prepared.append(dict(sampling_seed=seed,
                             metrics=summary,
                             factorial_and_method_contrasts=deltas,
                             factorial_main_and_interaction_effects=factorial,
                             clustering_diagnostics=record["diagnostics"],
                             timings_seconds=record["timings"],
                             per_case_metrics=sorted(rows,key=lambda r:(r["patient"],r["method"]))))
        all_seeds.append(seed)
    average={}
    for method in methods:
        average[method]={}
        for metric in METRICS:
            vals=[block["metrics"][method][metric]["mean"] for block in prepared]
            average[method][metric]=describe([v for v in vals if v is not None])
    mean_deltas={}
    for contrast in prepared[0]["factorial_and_method_contrasts"]:
        mean_deltas[contrast]={}
        for metric in METRICS:
            vals=[block["factorial_and_method_contrasts"][contrast][metric]["mean"]
                  for block in prepared]
            mean_deltas[contrast][metric]=describe([v for v in vals if v is not None])
    summary_factorial={}
    for effect in prepared[0]["factorial_main_and_interaction_effects"]:
        summary_factorial[effect]={}
        for metric in METRICS:
            means=[r["factorial_main_and_interaction_effects"][effect][metric]["mean"]
                   for r in prepared]
            summary_factorial[effect][metric]=describe([v for v in means if v is not None])
    report=dict(
        report_schema_version="3.0.0",
        status="completed",
        started_at_utc=started_at,
        completed_at_utc=utc_timestamp(),
        contains_images=False,
        contains_pixel_arrays=False,
        main_result_file="experiment_results.json",
        methods=list(methods),
        sampling_seeds=all_seeds,
        fixed_patient_split=dict(seed=split["seed"],train_count=len(split["train"]),
                                 test_count=len(split["test"]),
                                 train=sorted(split["train"]),test=sorted(split["test"])),
        metadata=metadata,
        overall_timings_seconds=common_timings,
        metrics_across_sampling_seeds=average,
        paired_contrasts_across_sampling_seeds=mean_deltas,
        factorial_effects_across_sampling_seeds=summary_factorial,
        per_sampling_seed=prepared,
        definitions=dict(
            metrics_across_sampling_seeds="Describes variation of each seed's PATIENT MACRO metric; n = number of seeds, not independent patients",
            paired_contrasts_across_sampling_seeds="Variability of mean within-patient method differences across bank-sampling seeds; not an independent-patient CI",
            full_body_metrics="Unprefixed metrics are full-image; body_* are restricted to CT HU body",
            body_coverage="body_excluded_positive_pixels computed only during post-hoc evaluation",
            top_area="top0p1/top0p5/top1/top5: predefined 0.1%/0.5%/1%/5% of evaluated ROI",
            local_support="Query-local kNN patient-concentration penalty; not global cluster-patient coverage",
            ablation="2x2 factorial random vs diverse, reference size 512 vs 8192; comparisons are descriptive, test data already seen in prior exploratory rounds",
            factorial_effects="bank_size_main averages 8192-512 over random/diverse; diversity_main averages diverse-random over sizes; interaction=(diverse8192-random8192)-(diverse512-random512)",
            random_seeds="Fixed patient split, multiple random bank subsamples; cannot claim independent external validation",
        ),
        limitations=[
            "Experiments are exploratory because this test split informed previous iterations; an untouched test set is required for confirmatory claims",
            "Only tumor-positive patients: cannot measure patient-level healthy-vs-cancer discrimination",
            "Independent TORC-inspired approximation; not proven MATLAB P-code parity",
            "Anatomically rare healthy structures may produce false positives",
            "No CT, masks, images, score grids or feature tensors in report",
            "No experiment results were uploaded or synchronized to GitHub",
        ])
    payload=clean_json(report)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".incomplete_multiseed_",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path
