from scripts.check_staged import blocked


def test_research_results_guard():
    assert blocked("outputs/full/experiment_results.json")
    assert blocked("experiment_results_2026.json")
    assert blocked("metrics/experiment_results.json")
    assert blocked("data/archives/Task10_Colon.tar")
    assert blocked("random/cache/case.npz")
    assert blocked("golden.mat")
    assert not blocked("lightweight_clustering/reporting.py")
    assert not blocked("tests/test_reporting.py")
    assert not blocked("docs/PROTOCOL.md")
