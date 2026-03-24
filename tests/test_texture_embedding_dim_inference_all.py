from tracking.classification.engine import _infer_texture_embedding_dim


def test_infer_texture_embedding_dims_for_all_pretrain_extractors_defaults():
    assert _infer_texture_embedding_dim("tab_v3_pro", {}, {}) == 10
    assert _infer_texture_embedding_dim("tab_v4", {}, {}) == 11
    assert _infer_texture_embedding_dim("tsc_v3_pro", {}, {}) == 3

    assert _infer_texture_embedding_dim("tab_v2", {}, {}) == 64
    assert _infer_texture_embedding_dim("tab_v2_extend", {}, {}) == 88
    assert _infer_texture_embedding_dim("tsc_v2", {}, {}) == 8
    assert _infer_texture_embedding_dim("tsc_v2_extend", {}, {}) == 8


def test_tex_pca_dim_override_for_tsc_v2_family():
    assert _infer_texture_embedding_dim("tsc_v2", {"tex_pca_dim": 6}, {}) == 6
    assert _infer_texture_embedding_dim("tsc_v2_extend", {"tex_pca_dim": 5}, {}) == 5
