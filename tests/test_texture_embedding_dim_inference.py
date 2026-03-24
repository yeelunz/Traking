from tracking.classification.engine import _infer_texture_embedding_dim


def test_tab_v4_embedding_dim_from_feature_extractor_definition():
    dim = _infer_texture_embedding_dim("tab_v4", params={}, tp_cfg={})
    assert dim == 11


def test_embedding_dim_override_has_priority():
    dim = _infer_texture_embedding_dim(
        "tab_v4",
        params={"texture_dim": 99},
        tp_cfg={"embedding_dim": 7},
    )
    assert dim == 7
