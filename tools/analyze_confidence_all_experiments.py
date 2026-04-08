from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(r"c:/Users/User/Desktop/code/Traking")
RESULTS_ROOT = WORKSPACE / "results"
DATASET_ROOTS = [
    WORKSPACE / "dataset" / "merged_extend",
    WORKSPACE / "dataset" / "merged",
]
OUT_DIR = WORKSPACE / "analysis" / "confidence_relation_all_experiments"
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
    n_perm: int = 1000,
    max_n: int = 50000,
    seed: int = 3407,
) -> float | None:
    obs = fn(x, y)
    if obs is None:
        return None
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        idx = rng.choice(len(x), size=max_n, replace=False)
        x = x[idx]
        y = y[idx]
        obs = fn(x, y)
        if obs is None:
            return None
    cnt = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        st = fn(x, yp)
        if st is None:
            continue
        if abs(st) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def _bootstrap_ci_diff_mean(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 3000,
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
    n_perm: int = 2000,
    max_n: int = 50000,
    seed: int = 3407,
) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), size=max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), size=max_n, replace=False)]
    obs = float(np.mean(a) - np.mean(b))
    all_v = np.concatenate([a, b])
    n_a = len(a)
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


def _corr_stats(scores: np.ndarray, values: np.ndarray, *, with_perm: bool) -> CorrResult:
    pearson = _pearson(scores, values)
    spearman = _spearman(scores, values)
    slope, intercept = _linreg(scores, values)
    p_pearson = _perm_pvalue(scores, values, _pearson) if with_perm else None
    p_spearman = _perm_pvalue(scores, values, _spearman) if with_perm else None
    return CorrResult(
        n=int(len(scores)),
        pearson_r=pearson,
        spearman_rho=spearman,
        slope=slope,
        intercept=intercept,
        perm_p_pearson=p_pearson,
        perm_p_spearman=p_spearman,
    )


def _load_gt_map(subject: str, video: str, cache: dict[tuple[str, str], dict[int, list[float]]]) -> dict[int, list[float]] | None:
    key = (subject, video)
    if key in cache:
        return cache[key]

    for root in DATASET_ROOTS:
        ann_path = root / subject / f"{video}.json"
        if not ann_path.exists():
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        id2f = {int(im["id"]): int(im["frame_index"]) for im in ann.get("images") or []}
        gt: dict[int, list[float]] = {}
        for a in ann.get("annotations") or []:
            img_id = int(a["image_id"])
            fi = id2f.get(img_id)
            if fi is not None:
                gt[fi] = list(map(float, a["bbox"]))
        cache[key] = gt
        return gt

    cache[key] = None
    return None


def _is_loso_name(experiment_rel: str) -> bool:
    return "loso" in experiment_rel.lower()


def _summarize_scored(scores: np.ndarray, iou: np.ndarray, ce: np.ndarray, *, with_perm: bool) -> dict[str, Any]:
    if len(scores) == 0:
        return {
            "n_scored": 0,
            "iou_corr": asdict(CorrResult(0, None, None, None, None, None, None)),
            "ce_corr": asdict(CorrResult(0, None, None, None, None, None, None)),
            "iou_low_high_20pct": None,
            "ce_low_high_20pct": None,
            "score_quantiles": None,
            "iou_mean": None,
            "ce_mean": None,
        }

    iou_corr = _corr_stats(scores, iou, with_perm=with_perm)
    ce_corr = _corr_stats(scores, ce, with_perm=with_perm)

    q20 = float(np.quantile(scores, 0.2))
    q80 = float(np.quantile(scores, 0.8))
    sel_low = scores <= q20
    sel_high = scores >= q80

    iou_low = iou[sel_low]
    iou_high = iou[sel_high]
    ce_low = ce[sel_low]
    ce_high = ce[sel_high]

    iou_delta = float(np.mean(iou_high) - np.mean(iou_low)) if len(iou_low) and len(iou_high) else None
    ce_delta = float(np.mean(ce_low) - np.mean(ce_high)) if len(ce_low) and len(ce_high) else None

    return {
        "n_scored": int(len(scores)),
        "iou_mean": float(np.mean(iou)),
        "ce_mean": float(np.mean(ce)),
        "score_quantiles": {"q20": q20, "q80": q80},
        "iou_corr": asdict(iou_corr),
        "ce_corr": asdict(ce_corr),
        "iou_low_high_20pct": {
            "n_low": int(len(iou_low)),
            "n_high": int(len(iou_high)),
            "iou_low_mean": float(np.mean(iou_low)) if len(iou_low) else None,
            "iou_high_mean": float(np.mean(iou_high)) if len(iou_high) else None,
            "delta_high_minus_low": iou_delta,
            "delta_ci95": None if len(iou_low) < 2 or len(iou_high) < 2 else _bootstrap_ci_diff_mean(iou_high, iou_low),
            "perm_p": None if (not with_perm or len(iou_low) < 2 or len(iou_high) < 2) else _perm_test_diff_mean(iou_high, iou_low),
        },
        "ce_low_high_20pct": {
            "n_low": int(len(ce_low)),
            "n_high": int(len(ce_high)),
            "ce_low_mean": float(np.mean(ce_low)) if len(ce_low) else None,
            "ce_high_mean": float(np.mean(ce_high)) if len(ce_high) else None,
            "delta_low_minus_high": ce_delta,
            "delta_ci95": None if len(ce_low) < 2 or len(ce_high) < 2 else _bootstrap_ci_diff_mean(ce_low, ce_high),
            "perm_p": None if (not with_perm or len(ce_low) < 2 or len(ce_high) < 2) else _perm_test_diff_mean(ce_low, ce_high),
        },
    }


def main() -> None:
    pred_files = [
        p for p in RESULTS_ROOT.rglob("YOLOv11.json") if "test/detection/predictions_by_video/" in p.as_posix()
    ]
    pred_files = sorted(pred_files)

    gt_cache: dict[tuple[str, str], dict[int, list[float]] | None] = {}
    all_rows: list[dict[str, Any]] = []
    skipped_no_gt: set[tuple[str, str]] = set()

    for pf in pred_files:
        # .../<experiment>/test/detection/predictions_by_video/<subject>/<video>/YOLOv11.json
        try:
            experiment_dir = pf.parents[5]
        except IndexError:
            continue
        experiment_rel = experiment_dir.relative_to(RESULTS_ROOT).as_posix()
        subject = pf.parent.parent.name
        video = pf.parent.name
        is_loso = _is_loso_name(experiment_rel)

        gt_map = _load_gt_map(subject, video, gt_cache)
        if not gt_map:
            skipped_no_gt.add((subject, video))
            continue

        preds = json.loads(pf.read_text(encoding="utf-8"))
        for r in preds:
            fi = int(r.get("frame_index", -1))
            gt = gt_map.get(fi)
            if gt is None:
                continue
            pb = np.array(r["bbox"], dtype=np.float64)
            gb = np.array(gt, dtype=np.float64)
            score = _to_float_or_none(r.get("score"))
            fallback = bool(r.get("fallback", False) or r.get("is_fallback", False))
            all_rows.append(
                {
                    "experiment": experiment_rel,
                    "subject": subject,
                    "video": video,
                    "frame_index": fi,
                    "is_loso": is_loso,
                    "score": score,
                    "fallback": fallback,
                    "iou": _bbox_iou(pb, gb),
                    "ce": _center_error(pb, gb),
                }
            )

    if not all_rows:
        raise RuntimeError("No matched rows found across experiments.")

    raw_csv = OUT_DIR / "confidence_iou_ce_all_rows.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "subject",
                "video",
                "frame_index",
                "is_loso",
                "score",
                "fallback",
                "iou",
                "ce",
            ],
        )
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    scored_rows = [r for r in all_rows if r["score"] is not None and not np.isnan(float(r["score"]))]

    def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=np.float64)

    overall_scores = _arr(scored_rows, "score")
    overall_iou = _arr(scored_rows, "iou")
    overall_ce = _arr(scored_rows, "ce")

    overall_summary = _summarize_scored(overall_scores, overall_iou, overall_ce, with_perm=True)

    loso_rows = [r for r in scored_rows if r["is_loso"]]
    non_loso_rows = [r for r in scored_rows if not r["is_loso"]]
    loso_summary = _summarize_scored(_arr(loso_rows, "score") if loso_rows else np.array([]), _arr(loso_rows, "iou") if loso_rows else np.array([]), _arr(loso_rows, "ce") if loso_rows else np.array([]), with_perm=True)
    non_loso_summary = _summarize_scored(_arr(non_loso_rows, "score") if non_loso_rows else np.array([]), _arr(non_loso_rows, "iou") if non_loso_rows else np.array([]), _arr(non_loso_rows, "ce") if non_loso_rows else np.array([]), with_perm=True)

    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        by_experiment.setdefault(r["experiment"], []).append(r)

    experiment_stats: list[dict[str, Any]] = []
    for exp_name, rows in sorted(by_experiment.items()):
        exp_scored = [r for r in rows if r["score"] is not None and not np.isnan(float(r["score"]))]
        if len(exp_scored) >= 3:
            s = _arr(exp_scored, "score")
            i = _arr(exp_scored, "iou")
            c = _arr(exp_scored, "ce")
            exp_summary = _summarize_scored(s, i, c, with_perm=False)
            iou_corr = exp_summary["iou_corr"]
            ce_corr = exp_summary["ce_corr"]
            iou_lh = exp_summary["iou_low_high_20pct"]
            ce_lh = exp_summary["ce_low_high_20pct"]
        else:
            iou_corr = asdict(CorrResult(len(exp_scored), None, None, None, None, None, None))
            ce_corr = asdict(CorrResult(len(exp_scored), None, None, None, None, None, None))
            iou_lh = None
            ce_lh = None

        experiment_stats.append(
            {
                "experiment": exp_name,
                "is_loso": _is_loso_name(exp_name),
                "n_rows": int(len(rows)),
                "n_scored": int(len(exp_scored)),
                "subjects": ",".join(sorted({str(r['subject']) for r in rows})),
                "videos": int(len({(str(r['subject']), str(r['video'])) for r in rows})),
                "iou_mean": None if not exp_scored else float(np.mean(_arr(exp_scored, "iou"))),
                "ce_mean": None if not exp_scored else float(np.mean(_arr(exp_scored, "ce"))),
                "pearson_iou": iou_corr["pearson_r"],
                "spearman_iou": iou_corr["spearman_rho"],
                "pearson_ce": ce_corr["pearson_r"],
                "spearman_ce": ce_corr["spearman_rho"],
                "delta_iou_high_minus_low": None if not iou_lh else iou_lh["delta_high_minus_low"],
                "delta_ce_low_minus_high": None if not ce_lh else ce_lh["delta_low_minus_high"],
            }
        )

    exp_csv = OUT_DIR / "experiment_level_stats.csv"
    with exp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(experiment_stats[0].keys()))
        w.writeheader()
        for r in experiment_stats:
            w.writerow(r)

    valid_exp_iou = [e["pearson_iou"] for e in experiment_stats if e["pearson_iou"] is not None]
    valid_exp_ce = [e["pearson_ce"] for e in experiment_stats if e["pearson_ce"] is not None]
    support_exp = [
        e
        for e in experiment_stats
        if e["pearson_iou"] is not None and e["pearson_ce"] is not None and e["pearson_iou"] > 0 and e["pearson_ce"] < 0
    ]

    summary = {
        "coverage": {
            "prediction_files_found": int(len(pred_files)),
            "experiments_found": int(len(by_experiment)),
            "rows_total": int(len(all_rows)),
            "rows_scored": int(len(scored_rows)),
            "rows_unscored": int(len(all_rows) - len(scored_rows)),
            "subjects": sorted({str(r["subject"]) for r in all_rows}),
            "videos": sorted({f"{r['subject']}/{r['video']}" for r in all_rows}),
            "skipped_subject_video_no_gt": sorted([f"{s}/{v}" for s, v in skipped_no_gt]),
        },
        "overall": overall_summary,
        "loso": loso_summary,
        "non_loso": non_loso_summary,
        "experiment_relation_distribution": {
            "n_experiments_with_corr": int(len(valid_exp_iou)),
            "pearson_iou_median": None if not valid_exp_iou else float(np.median(np.array(valid_exp_iou))),
            "pearson_iou_q25": None if not valid_exp_iou else float(np.quantile(np.array(valid_exp_iou), 0.25)),
            "pearson_iou_q75": None if not valid_exp_iou else float(np.quantile(np.array(valid_exp_iou), 0.75)),
            "pearson_ce_median": None if not valid_exp_ce else float(np.median(np.array(valid_exp_ce))),
            "pearson_ce_q25": None if not valid_exp_ce else float(np.quantile(np.array(valid_exp_ce), 0.25)),
            "pearson_ce_q75": None if not valid_exp_ce else float(np.quantile(np.array(valid_exp_ce), 0.75)),
            "n_support_hypothesis": int(len(support_exp)),
            "support_ratio": None if not valid_exp_iou else float(len(support_exp) / len(valid_exp_iou)),
        },
        "artifacts": {
            "all_rows_csv": str(raw_csv.as_posix()),
            "experiment_stats_csv": str(exp_csv.as_posix()),
        },
    }

    summary_json = OUT_DIR / "confidence_iou_ce_all_experiments_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# Confidence vs IoU / Center Error: All Experiments Audit")
    md.append("")
    md.append("## Coverage")
    md.append(f"- prediction files found: {summary['coverage']['prediction_files_found']}")
    md.append(f"- experiments included: {summary['coverage']['experiments_found']}")
    md.append(f"- matched rows: {summary['coverage']['rows_total']}")
    md.append(f"- scored rows: {summary['coverage']['rows_scored']}")
    md.append(f"- unscored rows: {summary['coverage']['rows_unscored']}")
    md.append("")
    md.append("## Overall (All Scored Rows)")
    md.append(f"- IoU Pearson r: {summary['overall']['iou_corr']['pearson_r']} (perm p={summary['overall']['iou_corr']['perm_p_pearson']})")
    md.append(f"- IoU Spearman rho: {summary['overall']['iou_corr']['spearman_rho']} (perm p={summary['overall']['iou_corr']['perm_p_spearman']})")
    md.append(f"- CE Pearson r: {summary['overall']['ce_corr']['pearson_r']} (perm p={summary['overall']['ce_corr']['perm_p_pearson']})")
    md.append(f"- CE Spearman rho: {summary['overall']['ce_corr']['spearman_rho']} (perm p={summary['overall']['ce_corr']['perm_p_spearman']})")
    md.append(f"- IoU delta high20-low20: {summary['overall']['iou_low_high_20pct']['delta_high_minus_low']} (p={summary['overall']['iou_low_high_20pct']['perm_p']})")
    md.append(f"- CE delta low20-high20: {summary['overall']['ce_low_high_20pct']['delta_low_minus_high']} (p={summary['overall']['ce_low_high_20pct']['perm_p']})")
    md.append("")
    md.append("## LOSO vs Non-LOSO")
    md.append(f"- LOSO n_scored: {summary['loso']['n_scored']}")
    md.append(f"- LOSO IoU Pearson: {summary['loso']['iou_corr']['pearson_r']} | CE Pearson: {summary['loso']['ce_corr']['pearson_r']}")
    md.append(f"- Non-LOSO n_scored: {summary['non_loso']['n_scored']}")
    md.append(f"- Non-LOSO IoU Pearson: {summary['non_loso']['iou_corr']['pearson_r']} | CE Pearson: {summary['non_loso']['ce_corr']['pearson_r']}")
    md.append("")
    md.append("## Experiment-Level Distribution")
    md.append(f"- experiments with valid correlations: {summary['experiment_relation_distribution']['n_experiments_with_corr']}")
    md.append(f"- median Pearson(score,IoU): {summary['experiment_relation_distribution']['pearson_iou_median']}")
    md.append(f"- median Pearson(score,CE): {summary['experiment_relation_distribution']['pearson_ce_median']}")
    md.append(f"- support ratio (Pearson IoU>0 and CE<0): {summary['experiment_relation_distribution']['support_ratio']}")
    md.append("")
    md.append("## Artifacts")
    md.append(f"- all rows CSV: {summary['artifacts']['all_rows_csv']}")
    md.append(f"- experiment stats CSV: {summary['artifacts']['experiment_stats_csv']}")
    md.append(f"- summary JSON: {summary_json.as_posix()}")

    report_md = OUT_DIR / "confidence_iou_ce_all_experiments_report.md"
    report_md.write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiments_found": summary["coverage"]["experiments_found"],
                "rows_total": summary["coverage"]["rows_total"],
                "rows_scored": summary["coverage"]["rows_scored"],
                "overall_iou_pearson": summary["overall"]["iou_corr"]["pearson_r"],
                "overall_ce_pearson": summary["overall"]["ce_corr"]["pearson_r"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
