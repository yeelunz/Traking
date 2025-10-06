import numpy as np
from tracking.preproc.srad import SRAD
from tracking.preproc.logdr import LogDynamicRange
from tracking.preproc.tgc import TGC

def _dummy_rgb(h=64,w=96):
    rng = np.random.default_rng(0)
    return (rng.random((h,w,3))*255).astype(np.uint8)

def _dummy_gray(h=64,w=96):
    rng = np.random.default_rng(1)
    return (rng.random((h,w))*255).astype(np.uint8)

def test_srad_shape_dtype():
    img = _dummy_rgb()
    proc = SRAD({"iterations":2}).apply_to_frame(img)
    assert proc.shape == img.shape
    assert proc.dtype == np.uint8

def test_logdr_log():
    img = _dummy_rgb()
    proc = LogDynamicRange({"method":"log"}).apply_to_frame(img)
    assert proc.shape == img.shape
    assert proc.dtype == np.uint8

def test_logdr_gamma():
    img = _dummy_rgb()
    proc = LogDynamicRange({"method":"gamma","gamma":0.8}).apply_to_frame(img)
    assert proc.shape == img.shape
    assert proc.dtype == np.uint8

def test_tgc_gray():
    g = _dummy_gray()
    proc = TGC({"mode":"linear","gain_end":1.5}).apply_to_frame(g)
    assert proc.shape == g.shape

def test_tgc_rgb():
    img = _dummy_rgb()
    proc = TGC({"mode":"exp","gain_end":3.0,"exp_k":2.0}).apply_to_frame(img)
    assert proc.shape == img.shape
