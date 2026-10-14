import numpy as np
import pytest
from lightweight_clustering.torc_approx import torque_approx, cluster_patient_support
from lightweight_clustering.experiment import metric_one, balanced_reference


def test_deterministic():
    rng=np.random.default_rng(42)
    x=np.r_[rng.normal(loc=-2,size=(12,3)),rng.normal(loc=2,size=(12,3))]
    a=torque_approx(x, fixed_k=2)
    b=torque_approx(x, fixed_k=2)
    assert np.array_equal(a.labels,b.labels)
    assert len(np.unique(a.labels))==2
    assert sum(a.cluster_sizes)==len(x)


def test_automatic_is_valid():
    x=np.random.default_rng(0).normal(size=(20,6))
    labels=torque_approx(x,max_clusters=6).labels
    assert len(labels)==20
    assert 2<=len(np.unique(labels))<=6


def test_support_counts_unique_patients():
    size,support=cluster_patient_support(
        np.array([0,0,0,1,1]), np.array(["a","a","b","b","c"]))
    assert np.isclose(support[0],2/3)
    assert np.isclose(size[0],3/5)


def test_reference_no_label_required():
    rng=np.random.default_rng(42)
    cases={str(i):(rng.normal(size=(2,3,3,4)).astype("float32"),np.arange(2)) for i in range(4)}
    x,p=balanced_reference(cases,{"0","1","2"},42,5,12)
    assert len(x)==len(p)==12
    assert set(p)<={"0","1","2"}


def test_metrics_edge_cases():
    null=metric_one(np.zeros((2,2)),np.zeros((2,2)))
    assert null["pixel_auroc"] is None
    assert null["dice_at_top5"] is None
    val=metric_one(np.array([[0,1],[2,3]]),np.array([[0,0],[0,1]]))
    assert val["pixel_auroc"]==1.0


def test_bad_matrix_guard():
    with pytest.raises(ValueError):
        torque_approx(np.ones((1025,3)))
