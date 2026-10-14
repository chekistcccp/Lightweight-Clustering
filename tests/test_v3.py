"""V3 local-patient support, factorial sampling and single report tests.

No real medical volumes, image outputs or patient experimental numbers are used.
"""
import json
import numpy as np
import pytest

from lightweight_clustering.local_support import local_patient_penalty, support_diagnostics
from lightweight_clustering.v3_reporting import compare_pairs,factorial_effects,write_multiseed_report,FACTORIAL_CONTRASTS
from lightweight_clustering.reporting import METRICS
from lightweight_clustering.sampling import collect_body_references
from lightweight_clustering.v3_experiment import parser,archive_previous_report


def test_local_patient_source_concentration_changes_when_global_clusters_same():
    owners=np.array([0,0,0,0,1,2,3,4,5,6,7,8])
    d=np.full((2,4),.15,dtype=np.float32)
    idx=np.array([[0,1,2,3],[4,5,6,7]])
    penalty=local_patient_penalty(d,idx,owners,temperature=.2)
    assert penalty[0]==pytest.approx(1.,abs=1e-6)
    assert penalty[1]==pytest.approx(0.,abs=1e-6)
    stat=support_diagnostics(penalty)
    assert stat["variation_detected"]
    assert stat["mean"]==pytest.approx(.5)


def test_local_support_uses_distance_weight_not_just_distinct_count():
    owner=np.arange(4)
    idx=np.array([[0,1,2,3],[0,1,2,3]])
    d=np.array([[.1,.1,.1,.1],[.01,.9,.9,.9]],dtype=np.float32)
    score=local_patient_penalty(d,idx,owner,temperature=.1)
    assert score[0]==pytest.approx(0.,abs=1e-6)
    assert score[1]>.95


def test_local_support_invalid_inputs():
    with pytest.raises(ValueError):
        local_patient_penalty(np.ones((2,1)),np.zeros((2,1),dtype=int),
                              np.array([0]),temperature=.1)
    with pytest.raises(ValueError):
        local_patient_penalty(np.ones((2,2)),np.zeros((2,2),dtype=int),
                              np.array([0]),temperature=0)


def test_factorial_prefix_controls_bank_capacity():
    rng=np.random.default_rng(3)
    cases={f"p{i}":(rng.normal(size=(3,5,4,6)).astype(np.float32),
                     np.arange(3)) for i in range(5)}
    body={patient:np.ones((3,5,4),dtype=bool) for patient in cases}
    for diverse in [False,True]:
        large,people,_=collect_body_references(cases,body,set(cases),42,
                  per_case=12,maximum=45,diverse=diverse,candidate_pool=20)
        assert len(large)==45
        small=large[:15];small_people=people[:15]
        assert len(small)==15 and len(np.unique(small_people))==5
        np.testing.assert_allclose(small,large[:15])


METHODS=("patch_knn","body_patch_knn","body_diverse_512",
         "body_random_8192","diverse_patch_knn","torc_approx",
         "local_consensus_knn","torc_consensus")


def synth_record(seed,base=.3):
    rows=[]
    for i,method in enumerate(METHODS):
        for patient in ("case1","case2"):
            row=dict(patient=patient,method=method,slices=1,
                     positive_pixels=2,negative_pixels=14,body_valid_pixels=12,
                     body_excluded_positive_pixels=0)
            row.update({m:base+i*.01+seed*.000001 for m in METRICS})
            row["pr_curve"]={"full":[{"precision":.1,"recall":.2}],
                             "body":[{"precision":.2,"recall":.3}]}
            rows.append(row)
    return dict(sampling_seed=seed,rows=rows,
                diagnostics={"torc_clusters":2,"local_support":{"q95":.5}},
                timings={"total":1.2})


def test_multiseed_report_has_all_ablation_contrasts_and_single_file(tmp_path):
    file=tmp_path/"outputs"/"full"/"experiment_results.json"
    kwargs=dict(target=file,methods=METHODS,
                records=[synth_record(42),synth_record(123)],
                split=dict(seed=42,train=["case3"],test=["case1","case2"]),
                metadata={"results_were_uploaded_to_github":False},
                common_timings={"overall_seconds":10},
                started_at="2026-10-14T00:00:00+00:00")
    write_multiseed_report(**kwargs)
    result=json.loads(file.read_text())
    assert result["report_schema_version"]=="3.0.0"
    assert result["status"]=="completed"
    assert result["contains_images"] is False
    assert result["sampling_seeds"]==[42,123]
    assert len(result["per_sampling_seed"])==2
    assert len(result["per_sampling_seed"][0]["per_case_metrics"])==len(METHODS)*2
    assert len(result["paired_contrasts_across_sampling_seeds"])==len(FACTORIAL_CONTRASTS)
    assert result["metrics_across_sampling_seeds"]["diverse_patch_knn"]["pixel_ap"]["n"]==2
    assert not (file.parent/"per_case.csv").exists()


def test_fail_without_all_seed_methods_and_without_duplicate_seeds(tmp_path):
    path=tmp_path/"experiment_results.json"
    kwargs=dict(target=path,methods=METHODS,
                records=[synth_record(42),synth_record(123)],
                split=dict(seed=42,train=["case3"],test=["case1","case2"]),
                metadata={},common_timings={},started_at="2026-10-14T00:00:00+00:00")
    kwargs["records"][1]["rows"].pop()
    with pytest.raises(ValueError,match="Incomplete"):
        write_multiseed_report(**kwargs)
    assert not path.exists()
    kwargs["records"]=[synth_record(42),synth_record(42)]
    with pytest.raises(ValueError,match="Duplicate sampling seed"):
        write_multiseed_report(**kwargs)


def test_no_raw_arrays_in_report(tmp_path):
    kwargs=dict(target=tmp_path/"experiment_results.json",methods=METHODS,
                records=[synth_record(42)],
                split=dict(seed=42,train=["case3"],test=["case1","case2"]),
                metadata={"score_grid":np.zeros((3,3))},
                common_timings={},started_at="2026-10-14T00:00:00+00:00")
    with pytest.raises(ValueError,match="Forbidden"):
        write_multiseed_report(**kwargs)


def test_v3_defaults_and_one_seed_override():
    args=parser().parse_args([])
    assert args.sampling_seeds==[42,123,2026]
    assert args.local_support_k>=args.knn
    assert args.save_predictions is False
    assert parser().parse_args(["--sampling-seeds","42"]).sampling_seeds==[42]


def test_local_archive_keeps_earlier_report(tmp_path):
    src=tmp_path/"outputs"/"full"/"experiment_results.json"
    src.parent.mkdir(parents=True)
    src.write_text('{"status":"completed"}',encoding="utf-8")
    archive_previous_report(src,tmp_path/"outputs")
    assert not src.exists()
    restored=list((tmp_path/"outputs"/"history").glob("**/experiment_results.json"))
    assert len(restored)==1
    assert json.loads(restored[0].read_text())["status"]=="completed"


def test_factorial_main_effects_and_interaction_are_patient_paired():
    rec=synth_record(42)
    vals={
        "body_patch_knn":0.10,
        "body_diverse_512":0.14,
        "body_random_8192":0.20,
        "diverse_patch_knn":0.28,
    }
    for row in rec["rows"]:
        if row["method"] in vals:
            row["pixel_ap"]=vals[row["method"]]
    effects=factorial_effects(rec["rows"])
    assert effects["bank_size_main"]["pixel_ap"]["mean"]==pytest.approx(.12)
    assert effects["diversity_main"]["pixel_ap"]["mean"]==pytest.approx(.06)
    assert effects["size_x_diversity_interaction"]["pixel_ap"]["mean"]==pytest.approx(.04)
