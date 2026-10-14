"""Synthetic tests for mask, patient sampling and direct AP/AUC (no medical images)."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score,roc_auc_score

from lightweight_clustering.body import ct_body_mask,patch_body_occupancy,body_patch_candidate_mask
from lightweight_clustering.sampling import (
    farthest_indices,collect_body_references,representative_torc_bank,
    propagate_cluster_labels,score_support,
)
from lightweight_clustering.metrics_v2 import rank_metrics,patient_metrics,numeric_fields


def test_body_mask_ignores_air_and_has_fractional_patch_support():
    yy,xx=np.ogrid[:128,:128]
    central=(yy-64)**2/40**2+(xx-64)**2/30**2<1
    hu=np.where(central,60.0,-1024.0).astype(np.float32)
    m=ct_body_mask(hu,size=64)
    assert m[32,32]
    assert not m[0,0]
    occupancy=patch_body_occupancy(m,8)
    assert occupancy.shape==(8,8)
    assert occupancy[4,4]>0.9
    assert occupancy[0,0]==0


def test_farthest_reproducible_patient_balanced_and_propagation():
    rng=np.random.default_rng(42)
    cases={f"p{i}":(rng.normal(size=(2,3,3,4)).astype(np.float32),
                     np.arange(2)) for i in range(5)}
    masks={p:np.ones((2,3,3),dtype=bool) for p in cases}
    a,p,counts=collect_body_references(cases,masks,set(cases),42,
                          per_case=6,maximum=20,diverse=True,candidate_pool=12)
    b,q,_=collect_body_references(cases,masks,set(cases),42,
                          per_case=6,maximum=20,diverse=True,candidate_pool=12)
    assert a.shape==(20,4)
    assert np.array_equal(a,b)
    assert np.array_equal(p,q)
    assert len(np.unique(p))==5
    assert set(counts.values())=={18}
    sub,own,ind=representative_torc_bank(a,p,maxpoints=10,seed=42,pool_limit=20)
    assert len(sub)==len(own)==len(ind)==10
    assert len(set(own))==5
    labels=np.array([0,0,0,0,1,1,1,1,1,1])
    full=propagate_cluster_labels(a,sub,labels)
    assert set(full)<=set(labels)
    size,cover=score_support(full,p)
    assert len(size)==len(cover)==len(a)
    assert np.all((cover>=0)&(cover<=1))


def test_metrics_match_sklearn_auroc_ap_with_ties():
    score=np.array([0.,.2,.2,.3,.5,.8,.8,.9],dtype=np.float32)
    gt=np.array([0,1,0,1,0,1,0,1],dtype=bool)
    m,curve=rank_metrics(score,gt)
    assert m["pixel_auroc"]==pytest.approx(roc_auc_score(gt,score),abs=1e-7)
    assert m["pixel_ap"]==pytest.approx(average_precision_score(gt,score),abs=1e-7)
    assert len(curve)==21
    assert m["tp_at_top0p1"]+m["fp_at_top0p1"]==1
    assert all(0<=p["precision"]<=1 for p in curve)


def test_body_roi_not_used_to_change_ground_truth():
    score=np.zeros((2,8,8),dtype=np.float32)
    gt=np.zeros_like(score,dtype=bool)
    gt[:,4,4]=True
    body=np.zeros_like(gt)
    body[:,:,3:7]=True
    score[:,4,4]=10
    row,curves=patient_metrics(score,gt,body)
    assert row["body_positive_pixels"]==2
    assert row["positive_pixels"]==2
    assert row["body_excluded_positive_pixels"]==0
    assert row["body_pixel_ap"]>=row["pixel_ap"]
    assert set(curves)=={"full","body"}
    assert "body_dice_at_top0p1" in row
    assert set(numeric_fields())>=set(["pixel_ap","body_pixel_ap",
                                       "dice_at_top0p1","body_dice_at_top5"])


def test_body_roi_tracks_excluded_lesions():
    score=np.arange(64,dtype=np.float32).reshape(8,8)
    gt=np.zeros((8,8),bool);gt[2,2]=1;gt[6,6]=1
    body=np.zeros((8,8),bool);body[:4]=1
    row,_=patient_metrics(score,gt,body)
    assert row["body_excluded_positive_pixels"]==1
    assert row["tumor_coverage_by_body"]==pytest.approx(0.5)


def test_chunked_exact_torch_cosine_knn_if_available():
    torch=pytest.importorskip("torch")
    from sklearn.metrics.pairwise import cosine_distances
    from lightweight_clustering.v2_experiment import knn_cosine
    rng=np.random.default_rng(111)
    bank=rng.normal(size=(23,9)).astype(np.float32)
    query=rng.normal(size=(11,9)).astype(np.float32)
    d,ix=knn_cosine(query,bank,k=4,device="cpu",chunk=3)
    reference=np.sort(cosine_distances(query,bank),axis=1)[:,:4]
    assert d.shape==(11,4)
    assert ix.shape==(11,4)
    np.testing.assert_allclose(d,reference,rtol=1e-5,atol=1e-6)
