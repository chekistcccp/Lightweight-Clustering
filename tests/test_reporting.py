"""No patient volumes/images are required to validate the reporting contract."""
import json
import numpy as np
import pytest
from lightweight_clustering.reporting import METRICS, clean_json, export_report, aggregate


def sample_row(method, patient, shift=0):
    return dict(method=method, patient=patient, slices=2,
                positive_pixels=3, negative_pixels=7,
                pixel_auroc=0.6+shift, pixel_ap=0.4+shift,
                dice_at_top5=0.3+shift, iou_at_top5=0.2+shift,
                precision_at_top5=0.2+shift, recall_at_top5=0.4+shift,
                fpr_at_top5=0.1-shift,top5_tp=2,top5_fp=1,top5_tn=6,top5_fn=1)


def base_report(tmp_path):
    methods=("patch_knn","torc_approx","torc_consensus")
    rows=[sample_row(method,p,shift=i*0.05)
          for i,method in enumerate(methods) for p in ("case_1","case_2")]
    file=tmp_path/"outputs"/"full"/"experiment_results.json"
    args=dict(methods=methods,rows=rows,metadata={"experiment_mode":"full", "git_commit":"abc"},
              diagnostics={"n_clusters":2,"cluster_counts":[3,4]},
              timings={"end_to_end_total_seconds":12.5},
              splits={"seed":42,"train":["case_3"],"test":["case_1","case_2"]},
              started_at="2026-10-14T12:00:00+00:00",
              completed_at="2026-10-14T12:00:12+00:00")
    return file,args


def test_all_metrics_in_single_file_without_images(tmp_path):
    file,args=base_report(tmp_path)
    assert export_report(file,**args)==file
    obj=json.loads(file.read_text(encoding="utf-8"))
    assert obj["status"]=="completed"
    assert obj["contains_images"] is False
    assert obj["contains_pixel_arrays"] is False
    assert set(obj["metrics"])==set(args["methods"])
    for m in args["methods"]:
        assert set(METRICS).issubset(obj["metrics"][m].keys())
    assert len(obj["per_case_metrics"])==6
    assert obj["cluster_diagnostics"]["cluster_counts"]==[3,4]
    assert obj["timings_seconds"]["end_to_end_total_seconds"]==12.5
    assert "paired_deltas_vs_patch_knn" in obj
    assert not (file.parent/"per_case.csv").exists()
    assert not (file.parent/"summary.json").exists()
    assert "score_grid" not in file.read_text()
    assert "heatmap" not in json.dumps(obj["per_case_metrics"])


def test_no_result_written_on_incomplete_experiment(tmp_path):
    file,args=base_report(tmp_path)
    args["rows"]=args["rows"][:-1]
    with pytest.raises(ValueError,match="Missing methods"):
        export_report(file,**args)
    assert not file.exists()


def test_reject_raw_image_array(tmp_path):
    file,args=base_report(tmp_path)
    args["diagnostics"]={"score_grid":np.zeros((10,10))}
    with pytest.raises(ValueError):
        export_report(file,**args)
    assert not file.exists()


def test_json_nonfinite_sanitized():
    obj=clean_json({"a":np.float32(float("nan")), "b":np.int64(2),
                    "c":np.float64(float("inf"))})
    assert obj=={"a":None,"b":2,"c":None}


def test_macros_and_paired_deltas(tmp_path):
    file,args=base_report(tmp_path)
    export_report(file,**args)
    data=json.loads(file.read_text())
    m=data["metrics"]["patch_knn"]["pixel_auroc"]
    assert m["mean"]==pytest.approx(0.6)
    paired=data["paired_deltas_vs_patch_knn"]["torc_consensus_minus_patch_knn"]
    assert paired["pixel_auroc"]["mean"]==pytest.approx(0.1)


def test_report_filename_fixed(tmp_path):
    _,args=base_report(tmp_path)
    with pytest.raises(ValueError,match="filename"):
        export_report(tmp_path/"results.json",**args)
