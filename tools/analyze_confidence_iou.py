from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RESULT_DIR = Path(
	r"c:/Users/User/Desktop/code/Traking/results/replay_n003_full_rf_concat_debug_v2/2026-03-24_20-09-17_mednext_clahe_global_tabv4_rf_concat_loso_n003"
)
PRED_ROOT = RESULT_DIR / "test" / "detection" / "predictions_by_video" / "n003"
ANN_ROOT = Path(r"c:/Users/User/Desktop/code/Traking/dataset/merged_extend/n003")
OUT_DIR = RESULT_DIR / "analysis" / "confidence_iou_audit" / "n003"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CorrResult:
	n: int
	pearson_r: float | None
	spearman_rho: float | None
	slope: float | None
	intercept: float | None
	perm_p_pearson: float | None
	perm_p_spearman: float | None


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
	ax1, ay1, aw, ah = map(float, a)
	bx1, by1, bw, bh = map(float, b)
	ax2, ay2 = ax1 + aw, ay1 + ah
	bx2, by2 = bx1 + bw, by1 + bh
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
	inter = iw * ih
	union = aw * ah + bw * bh - inter
	return float(inter / union) if union > 0 else 0.0


def _center_error(a: np.ndarray, b: np.ndarray) -> float:
	acx = float(a[0] + a[2] / 2.0)
	acy = float(a[1] + a[3] / 2.0)
	bcx = float(b[0] + b[2] / 2.0)
	bcy = float(b[1] + b[3] / 2.0)
	return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _average_rank(values: np.ndarray) -> np.ndarray:
	order = np.argsort(values, kind="mergesort")
	sorted_vals = values[order]
	ranks = np.empty(len(values), dtype=np.float64)
	i = 0
	while i < len(values):
		j = i + 1
		while j < len(values) and sorted_vals[j] == sorted_vals[i]:
			j += 1
		avg_rank = (i + j - 1) / 2.0 + 1.0
		ranks[order[i:j]] = avg_rank
		i = j
	return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
	if len(x) < 3:
		return None
	x_std = float(np.std(x))
	y_std = float(np.std(y))
	if x_std < 1e-12 or y_std < 1e-12:
		return None
	return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
	if len(x) < 3:
		return None
	xr = _average_rank(x)
	yr = _average_rank(y)
	return _pearson(xr, yr)


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
	if len(x) < 3:
		return None, None
	x_var = float(np.var(x))
	if x_var < 1e-12:
		return None, None
	x_mean = float(np.mean(x))
	y_mean = float(np.mean(y))
	slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
	intercept = float(y_mean - slope * x_mean)
	return slope, intercept


def _perm_pvalue(
	x: np.ndarray,
	y: np.ndarray,
	fn,
	*,
	n_perm: int = 5000,
	seed: int = 3407,
) -> float | None:
	obs = fn(x, y)
	if obs is None:
		return None
	rng = np.random.default_rng(seed)
	cnt = 0
	for _ in range(n_perm):
		yp = y[rng.permutation(len(y))]
		st = fn(x, yp)
		if st is None:
			continue
		if abs(st) >= abs(obs):
			cnt += 1
	return float((cnt + 1) / (n_perm + 1))


def _bootstrap_ci_mean(
	values: np.ndarray,
	*,
	n_boot: int = 5000,
	alpha: float = 0.05,
	seed: int = 3407,
) -> tuple[float, float] | None:
	if len(values) < 2:
		return None
	rng = np.random.default_rng(seed)
	boots = np.empty(n_boot, dtype=np.float64)
	n = len(values)
	for i in range(n_boot):
		idx = rng.integers(0, n, size=n)
		boots[i] = float(np.mean(values[idx]))
	lo = float(np.quantile(boots, alpha / 2.0))
	hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
	return lo, hi


def _bootstrap_ci_diff_mean(
	a: np.ndarray,
	b: np.ndarray,
	*,
	n_boot: int = 5000,
	alpha: float = 0.05,
	seed: int = 3407,
) -> tuple[float, float] | None:
	if len(a) < 2 or len(b) < 2:
		return None
	rng = np.random.default_rng(seed)
	boots = np.empty(n_boot, dtype=np.float64)
	na = len(a)
	nb = len(b)
	for i in range(n_boot):
		ia = rng.integers(0, na, size=na)
		ib = rng.integers(0, nb, size=nb)
		boots[i] = float(np.mean(a[ia]) - np.mean(b[ib]))
	lo = float(np.quantile(boots, alpha / 2.0))
	hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
	return lo, hi


def _perm_test_diff_mean(
	a: np.ndarray,
	b: np.ndarray,
	*,
	n_perm: int = 10000,
	seed: int = 3407,
) -> float | None:
	if len(a) < 2 or len(b) < 2:
		return None
	obs = float(np.mean(a) - np.mean(b))
	all_v = np.concatenate([a, b])
	n_a = len(a)
	rng = np.random.default_rng(seed)
	cnt = 0
	for _ in range(n_perm):
		perm = all_v[rng.permutation(len(all_v))]
		d = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
		if abs(d) >= abs(obs):
			cnt += 1
	return float((cnt + 1) / (n_perm + 1))


def _to_float_or_none(v: Any) -> float | None:
	if v is None:
		return None
	try:
		return float(v)
	except Exception:
		return None


def _corr_stats(scores: np.ndarray, ious: np.ndarray) -> CorrResult:
	pearson = _pearson(scores, ious)
	spearman = _spearman(scores, ious)
	slope, intercept = _linreg(scores, ious)
	p_pearson = _perm_pvalue(scores, ious, _pearson)
	p_spearman = _perm_pvalue(scores, ious, _spearman)
	return CorrResult(
		n=int(len(scores)),
		pearson_r=pearson,
		spearman_rho=spearman,
		slope=slope,
		intercept=intercept,
		perm_p_pearson=p_pearson,
		perm_p_spearman=p_spearman,
	)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
	qs = np.linspace(0.0, 1.0, n_bins + 1)
	edges = np.quantile(values, qs)
	# avoid empty bins caused by duplicate edges
	edges = np.unique(edges)
	if len(edges) < 2:
		return np.array([values.min(), values.max() + 1e-9], dtype=np.float64)
	edges[-1] += 1e-12
	return edges.astype(np.float64)


def _bin_summary(scores: np.ndarray, ious: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
	edges = _quantile_bins(scores, n_bins)
	out: list[dict[str, Any]] = []
	for i in range(len(edges) - 1):
		lo = float(edges[i])
		hi = float(edges[i + 1])
		sel = (scores >= lo) & (scores < hi)
		if not np.any(sel):
			continue
		vals = ious[sel]
		ci = _bootstrap_ci_mean(vals)
		out.append(
			{
				"bin_index": i,
				"score_lo": lo,
				"score_hi": hi,
				"n": int(sel.sum()),
				"iou_mean": float(np.mean(vals)),
				"iou_median": float(np.median(vals)),
				"iou_std": float(np.std(vals, ddof=0)),
				"iou_ci95_lo": None if ci is None else float(ci[0]),
				"iou_ci95_hi": None if ci is None else float(ci[1]),
				"bad_rate_iou_lt_05": float(np.mean(vals < 0.5)),
			}
		)
	return out


def _load_gt_map(video: str) -> dict[int, list[float]]:
	ann = json.loads((ANN_ROOT / f"{video}.json").read_text(encoding="utf-8"))
	id2f = {int(im["id"]): int(im["frame_index"]) for im in ann.get("images") or []}
	gt: dict[int, list[float]] = {}
	for a in ann.get("annotations") or []:
		img_id = int(a["image_id"])
		if img_id in id2f:
			gt[id2f[img_id]] = list(map(float, a["bbox"]))
	return gt


def _iter_rows() -> Iterable[dict[str, Any]]:
	videos = sorted([p.name for p in PRED_ROOT.iterdir() if p.is_dir()])
	for video in videos:
		gt_map = _load_gt_map(video)
		preds = json.loads((PRED_ROOT / video / "YOLOv11.json").read_text(encoding="utf-8"))
		for r in preds:
			fi = int(r["frame_index"])
			gt = gt_map.get(fi)
			if gt is None:
				continue
			pb = np.array(r["bbox"], dtype=np.float64)
			gb = np.array(gt, dtype=np.float64)
			score = _to_float_or_none(r.get("score"))
			fallback = bool(r.get("fallback", False) or r.get("is_fallback", False))
			yield {
				"video": video,
				"frame_index": fi,
				"score": score,
				"fallback": fallback,
				"iou": _bbox_iou(pb, gb),
				"ce": _center_error(pb, gb),
			}


def main() -> None:
	rows = list(_iter_rows())
	if not rows:
		raise RuntimeError("No matched prediction/GT rows found.")

	# Save raw merged table for auditability.
	table_csv = OUT_DIR / "confidence_iou_table.csv"
	with table_csv.open("w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["video", "frame_index", "score", "fallback", "iou", "ce"])
		w.writeheader()
		for r in rows:
			w.writerow(r)

	videos = sorted({r["video"] for r in rows})
	iou_all = np.array([float(r["iou"]) for r in rows], dtype=np.float64)

	# Scored subset only (the true confidence signal).
	scored_rows = [r for r in rows if r["score"] is not None and not np.isnan(float(r["score"]))]
	scores = np.array([float(r["score"]) for r in scored_rows], dtype=np.float64)
	iou_scored = np.array([float(r["iou"]) for r in scored_rows], dtype=np.float64)

	corr_all_scored = _corr_stats(scores, iou_scored)
	bins_all_scored = _bin_summary(scores, iou_scored, n_bins=10)

	# Bottom vs top confidence group comparison (20% vs 20%).
	q20 = float(np.quantile(scores, 0.2))
	q80 = float(np.quantile(scores, 0.8))
	low = iou_scored[scores <= q20]
	high = iou_scored[scores >= q80]
	low_high_summary = {
		"q20": q20,
		"q80": q80,
		"n_low": int(len(low)),
		"n_high": int(len(high)),
		"iou_low_mean": float(np.mean(low)) if len(low) else None,
		"iou_high_mean": float(np.mean(high)) if len(high) else None,
		"delta_high_minus_low": float(np.mean(high) - np.mean(low)) if len(low) and len(high) else None,
		"delta_ci95": (
			None
			if len(low) < 2 or len(high) < 2
			else _bootstrap_ci_diff_mean(high, low)
		),
		"perm_p": None if len(low) < 2 or len(high) < 2 else _perm_test_diff_mean(high, low),
	}

	# Fallback vs non-fallback (not confidence score itself, but quality signal audit).
	fb = np.array([float(r["iou"]) for r in rows if r["fallback"]], dtype=np.float64)
	nfb = np.array([float(r["iou"]) for r in rows if not r["fallback"]], dtype=np.float64)
	fallback_cmp = {
		"n_fallback": int(len(fb)),
		"n_non_fallback": int(len(nfb)),
		"iou_fallback_mean": float(np.mean(fb)) if len(fb) else None,
		"iou_non_fallback_mean": float(np.mean(nfb)) if len(nfb) else None,
		"delta_non_fallback_minus_fallback": float(np.mean(nfb) - np.mean(fb)) if len(fb) and len(nfb) else None,
		"delta_ci95": (
			None
			if len(fb) < 2 or len(nfb) < 2
			else _bootstrap_ci_diff_mean(nfb, fb)
		),
		"perm_p": None if len(fb) < 2 or len(nfb) < 2 else _perm_test_diff_mean(nfb, fb),
	}

	# Per-video scored analysis.
	per_video: dict[str, Any] = {}
	for v in videos:
		v_rows = [r for r in scored_rows if r["video"] == v]
		if len(v_rows) < 3:
			per_video[v] = {"n_scored": len(v_rows), "note": "insufficient scored samples"}
			continue
		vx = np.array([float(r["score"]) for r in v_rows], dtype=np.float64)
		vy = np.array([float(r["iou"]) for r in v_rows], dtype=np.float64)
		per_video[v] = {
			"n_scored": int(len(v_rows)),
			"corr": asdict(_corr_stats(vx, vy)),
			"bins": _bin_summary(vx, vy, n_bins=min(5, max(3, len(v_rows) // 6))),
		}

	summary = {
		"scope": {
			"result_dir": str(RESULT_DIR.as_posix()),
			"subject": "n003",
			"videos": videos,
		},
		"counts": {
			"rows_total": int(len(rows)),
			"rows_scored": int(len(scored_rows)),
			"rows_unscored": int(len(rows) - len(scored_rows)),
		},
		"overall": {
			"iou_mean_all_rows": float(np.mean(iou_all)),
			"iou_mean_scored_rows": float(np.mean(iou_scored)),
			"corr_scored": asdict(corr_all_scored),
			"score_iou_bins_scored": bins_all_scored,
			"low_high_scored_20pct": low_high_summary,
			"fallback_vs_non_fallback": fallback_cmp,
		},
		"per_video": per_video,
		"artifacts": {
			"table_csv": str(table_csv.as_posix()),
		},
	}

	summary_json = OUT_DIR / "confidence_iou_summary.json"
	summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	# Compact markdown report for quick read.
	md_lines = []
	md_lines.append("# Confidence vs IoU Statistical Audit (n003)")
	md_lines.append("")
	md_lines.append(f"- Total matched rows: {len(rows)}")
	md_lines.append(f"- Scored rows: {len(scored_rows)}")
	md_lines.append(f"- Unscored rows: {len(rows) - len(scored_rows)}")
	md_lines.append("")
	md_lines.append("## Overall (Scored Rows)")
	md_lines.append(
		f"- Pearson r: {corr_all_scored.pearson_r} (perm p={corr_all_scored.perm_p_pearson})"
	)
	md_lines.append(
		f"- Spearman rho: {corr_all_scored.spearman_rho} (perm p={corr_all_scored.perm_p_spearman})"
	)
	md_lines.append(f"- Linear slope (IoU~score): {corr_all_scored.slope}")
	md_lines.append("")
	md_lines.append("## 20% Low vs 20% High Confidence")
	md_lines.append(f"- q20={q20}, q80={q80}")
	md_lines.append(f"- mean IoU low={low_high_summary['iou_low_mean']}, high={low_high_summary['iou_high_mean']}")
	md_lines.append(f"- delta(high-low)={low_high_summary['delta_high_minus_low']}")
	md_lines.append(f"- delta 95% CI={low_high_summary['delta_ci95']}")
	md_lines.append(f"- permutation p={low_high_summary['perm_p']}")
	md_lines.append("")
	md_lines.append("## Fallback vs Non-fallback")
	md_lines.append(
		f"- mean IoU fallback={fallback_cmp['iou_fallback_mean']}, non-fallback={fallback_cmp['iou_non_fallback_mean']}"
	)
	md_lines.append(
		f"- delta(non-fallback - fallback)={fallback_cmp['delta_non_fallback_minus_fallback']}"
	)
	md_lines.append(f"- permutation p={fallback_cmp['perm_p']}")

	(OUT_DIR / "confidence_iou_report.md").write_text("\n".join(md_lines), encoding="utf-8")

	print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
