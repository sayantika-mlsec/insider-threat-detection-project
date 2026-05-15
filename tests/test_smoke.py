def test_package_imports():
    from src import InsiderThreatDetector, fuse_feature_matrices
    det = InsiderThreatDetector()
    assert det.is_fitted is False