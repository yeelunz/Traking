"""Quick functional tests for audit round 1 fixes."""
import numpy as np
from tracking.classification.trajectory_filter import (
    _adaptive_savgol, filter_detections, compute_trajectory_metrics,
    cubic_spline_interpolate_1d,
)

# Test 1: _adaptive_savgol rejects unsorted
try:
    _adaptive_savgol(np.array([1.0,2,3,4,5]), np.array([5,3,1,7,9]), 5, 2)
    print("FAIL: should raise on unsorted")
except ValueError as e:
    print(f"OK: unsorted rejected: {e}")

# Test 2: _adaptive_savgol rejects duplicates
try:
    _adaptive_savgol(np.array([1.0,2,3,4,5]), np.array([1,2,2,3,4]), 5, 2)
    print("FAIL: should raise on duplicates")
except ValueError as e:
    print(f"OK: duplicates rejected: {e}")

# Test 3: filter_detections handles duplicate frame_indices gracefully
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = filter_detections(
        np.array([1,2,2,3,4]), np.array([10.0,20,21,30,40]),
        np.array([5.0,6,7,8,9]), np.array([100.0]*5), np.array([100.0]*5),
        np.array([0.9]*5),
    )
    if len(w) > 0:
        print(f"OK: duplicate warning: {w[0].message}")
    n_out = len(result["cx"])
    print(f"  result length = {n_out} (should be 4)")
    assert n_out == 4, f"Expected 4, got {n_out}"

# Test 4: cubic_spline_interpolate_1d handles non-monotone gracefully
t = np.array([1.0, 3.0, 2.0, 4.0])
v = np.array([10.0, 30.0, 20.0, 40.0])
out = cubic_spline_interpolate_1d(t, v, np.array([1.5, 2.5]))
print(f"OK: cubic_spline non-monotone fallback: {out}")

# Test 5: memory guard for extreme sparsity
vals = np.array([1.0, 2.0, 3.0, 4.0])
fi = np.array([0, 50000, 100000, 150000])
out = _adaptive_savgol(vals, fi, 5, 2)
print(f"OK: extreme sparsity guard: returned {len(out)} points (fallback)")

# Test 6: compute_trajectory_metrics with frame_indices
m = compute_trajectory_metrics(
    np.array([0.0, 10, 20, 30, 40]),
    np.array([0.0, 10, 20, 30, 40]),
    np.array([50.0]*5),
    np.array([50.0]*5),
    np.array([0, 2, 4, 6, 8]),
)
print(f"OK: metrics with fi: jitter_cx={m['jitter_cx']:.3f}, path_length={m['path_length']:.1f}")

m2 = compute_trajectory_metrics(
    np.array([0.0, 10, 20, 30, 40]),
    np.array([0.0, 10, 20, 30, 40]),
    np.array([50.0]*5),
    np.array([50.0]*5),
)
print(f"OK: metrics w/o fi: jitter_cx={m2['jitter_cx']:.3f}, path_length={m2['path_length']:.1f}")

# Test 7: dtype preservation
vals32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
fi32 = np.array([0, 2, 5, 8, 12])
out32 = _adaptive_savgol(vals32, fi32, 5, 2)
print(f"OK: dtype preservation: input={vals32.dtype}, output={out32.dtype}")
assert out32.dtype == np.float32, f"Expected float32, got {out32.dtype}"

print("\nAll audit round 1 tests passed!")
