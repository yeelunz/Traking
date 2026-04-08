from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

# Use non-interactive backend so the script is runnable on headless CI/local shells.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.classification.trajectory_filter import _adaptive_savgol, hampel_then_pchip_1d


N_FRAMES = 200


@dataclass(frozen=True)
class FilterConfig:
    macro_ratio: float = 0.06
    macro_sigma: float = 2.0
    micro_hw: int = 2
    micro_sigma: float = 2.0
    max_outlier_run: int = 1
    sg_window: int = 5
    sg_polyorder: int = 2
    anchor_keep_ratio: float = 0.35


def _trajectory_a(n: int = N_FRAMES) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 + 180.0 * t
    y = 20.0 + 90.0 * t
    return np.column_stack([x, y]).astype(np.float64)


def _trajectory_b(n: int = N_FRAMES) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 + 180.0 * t
    # Smooth S-like path with mixed frequencies.
    y = 65.0 + 30.0 * np.sin(2.0 * np.pi * t) + 8.0 * np.sin(4.0 * np.pi * t + 0.5)
    return np.column_stack([x, y]).astype(np.float64)


def _trajectory_c(n: int = N_FRAMES) -> np.ndarray:
    # Sharp V turn around the middle frame.
    t = np.arange(n, dtype=np.float64)
    x = 10.0 + 0.9 * t
    apex = (n - 1) / 2.0
    y = 20.0 + 0.95 * np.abs(t - apex)
    return np.column_stack([x, y]).astype(np.float64)


def build_gt_library() -> dict[str, np.ndarray]:
    return {
        "A_line": _trajectory_a(),
        "B_s_curve": _trajectory_b(),
        "C_v_turn": _trajectory_c(),
    }


def add_jitter(gt: np.ndarray, rng: np.random.Generator, sigma: float = 0.7) -> np.ndarray:
    return gt + rng.normal(0.0, sigma, size=gt.shape)


def inject_pattern(
    normal: np.ndarray,
    pattern: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    raw = normal.copy()
    n = len(raw)
    missing_mask = np.zeros(n, dtype=bool)
    injected_outlier_mask = np.zeros(n, dtype=bool)

    if pattern == "P1_swiss_cheese":
        seg_len = int(rng.integers(20, 41))
        start = int(rng.integers(10, n - seg_len - 10))
        idx = np.arange(start, start + seg_len)
        missing_mask[idx[1::2]] = True
    elif pattern == "P2_long_occlusion":
        seg_len = int(rng.integers(5, 16))
        start = int(rng.integers(10, n - seg_len - 10))
        missing_mask[start : start + seg_len] = True
    elif pattern == "P3_isolated_spikes":
        k = int(rng.integers(1, 3))
        idx = rng.choice(np.arange(8, n - 8), size=k, replace=False)
        injected_outlier_mask[idx] = True
    elif pattern == "P4_consecutive_spikes":
        run_len = int(rng.integers(3, 6))
        start = int(rng.integers(10, n - run_len - 10))
        injected_outlier_mask[start : start + run_len] = True
    elif pattern == "P5_clean_only_jitter":
        pass
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    if injected_outlier_mask.any():
        angles = rng.uniform(0.0, 2.0 * np.pi, size=int(injected_outlier_mask.sum()))
        magnitudes = rng.uniform(35.0, 65.0, size=int(injected_outlier_mask.sum()))
        offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * magnitudes[:, None]
        raw[injected_outlier_mask] += offsets

    raw[missing_mask] = np.nan
    info: dict[str, np.ndarray | int] = {
        "missing_mask": missing_mask,
        "injected_outlier_mask": injected_outlier_mask,
        "num_missing": int(missing_mask.sum()),
        "num_injected_outliers": int(injected_outlier_mask.sum()),
    }
    return raw, info


def run_filter_pipeline(raw_xy: np.ndarray, cfg: FilterConfig) -> dict[str, np.ndarray]:
    n = len(raw_xy)
    frame_idx = np.arange(n, dtype=np.int64)
    observed_mask = np.isfinite(raw_xy).all(axis=1)

    filled = np.zeros_like(raw_xy, dtype=np.float64)
    marked = np.zeros_like(raw_xy, dtype=np.float64)
    outlier_mask_xy = np.zeros((n, 2), dtype=bool)

    params = {
        "macro_ratio": float(cfg.macro_ratio),
        "macro_sigma": float(cfg.macro_sigma),
        "micro_hw": int(cfg.micro_hw),
        "micro_sigma": float(cfg.micro_sigma),
        "max_outlier_run": int(cfg.max_outlier_run),
    }

    for d in range(2):
        values = np.asarray(raw_xy[:, d], dtype=np.float64)
        filled_d, marked_d, outlier_mask_d = hampel_then_pchip_1d(
            values,
            frame_idx,
            observed_mask=observed_mask,
            **params,
        )
        filled[:, d] = filled_d
        marked[:, d] = marked_d
        outlier_mask_xy[:, d] = outlier_mask_d

    smoothed = np.zeros_like(raw_xy, dtype=np.float64)
    for d in range(2):
        smoothed[:, d] = _adaptive_savgol(
            filled[:, d],
            frame_idx,
            window_length=int(cfg.sg_window),
            polyorder=int(cfg.sg_polyorder),
        )

    # Do-no-harm blend: keep trusted detector points closer to raw.
    flagged_mask = (~observed_mask) | outlier_mask_xy.any(axis=1)
    trusted_mask = (~flagged_mask) & observed_mask
    final = smoothed.copy()
    keep = float(np.clip(cfg.anchor_keep_ratio, 0.0, 1.0))
    if keep > 0.0 and trusted_mask.any():
        final[trusted_mask] = keep * raw_xy[trusted_mask] + (1.0 - keep) * final[trusted_mask]

    return {
        "filtered": final,
        "filled": filled,
        "marked": marked,
        "outlier_mask_any": outlier_mask_xy.any(axis=1),
        "observed_mask": observed_mask,
        "flagged_mask": flagged_mask,
        "trusted_mask": trusted_mask,
    }


def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def evaluate_one_case(
    gt: np.ndarray,
    raw: np.ndarray,
    pattern_info: dict[str, np.ndarray | int],
    out: dict[str, np.ndarray],
) -> dict[str, float | int]:
    filtered = out["filtered"]
    observed_mask = out["observed_mask"]
    flagged_mask = out["flagged_mask"]

    missing_mask = np.asarray(pattern_info["missing_mask"], dtype=bool)
    injected_outlier_mask = np.asarray(pattern_info["injected_outlier_mask"], dtype=bool)

    normal_mask = (~missing_mask) & (~injected_outlier_mask) & observed_mask
    moved_normal_mask = normal_mask & (_euclidean(filtered, np.nan_to_num(raw, nan=0.0)) > 1e-9)

    filtered_err = _euclidean(filtered, gt)

    raw_valid_err = _euclidean(raw[observed_mask], gt[observed_mask]) if observed_mask.any() else np.array([], dtype=np.float64)
    normal_raw_err = _euclidean(raw[normal_mask], gt[normal_mask]) if normal_mask.any() else np.array([], dtype=np.float64)
    normal_filtered_err = _euclidean(filtered[normal_mask], gt[normal_mask]) if normal_mask.any() else np.array([], dtype=np.float64)
    moved_normal_filtered_err = _euclidean(filtered[moved_normal_mask], gt[moved_normal_mask]) if moved_normal_mask.any() else np.array([], dtype=np.float64)

    harmful = (
        (normal_filtered_err - normal_raw_err) > 1e-6
        if normal_mask.any()
        else np.array([], dtype=bool)
    )

    corrupted_observed = injected_outlier_mask & observed_mask
    corr_raw_err = _euclidean(raw[corrupted_observed], gt[corrupted_observed]) if corrupted_observed.any() else np.array([], dtype=np.float64)
    corr_filtered_err = _euclidean(filtered[corrupted_observed], gt[corrupted_observed]) if corrupted_observed.any() else np.array([], dtype=np.float64)

    return {
        "n_frames": int(len(gt)),
        "n_missing": int(missing_mask.sum()),
        "n_injected_outliers": int(injected_outlier_mask.sum()),
        "n_flagged_by_filter": int(flagged_mask.sum()),
        "raw_observed_mean_err": float(raw_valid_err.mean()) if raw_valid_err.size else float("nan"),
        "filtered_mean_err": float(filtered_err.mean()),
        "normal_raw_mean_err": float(normal_raw_err.mean()) if normal_raw_err.size else float("nan"),
        "normal_filtered_mean_err": float(normal_filtered_err.mean()) if normal_filtered_err.size else float("nan"),
        "normal_err_delta": float((normal_filtered_err - normal_raw_err).mean()) if normal_raw_err.size else 0.0,
        "normal_harm_rate": float(harmful.mean()) if harmful.size else 0.0,
        "moved_normal_mean_err": float(moved_normal_filtered_err.mean()) if moved_normal_filtered_err.size else float("nan"),
        "corrupted_raw_mean_err": float(corr_raw_err.mean()) if corr_raw_err.size else float("nan"),
        "corrupted_filtered_mean_err": float(corr_filtered_err.mean()) if corr_filtered_err.size else float("nan"),
        "corrupted_gain": float((corr_raw_err - corr_filtered_err).mean()) if corr_raw_err.size else 0.0,
    }


def aggregate_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "filtered_mean_err": _mean("filtered_mean_err"),
        "raw_observed_mean_err": _mean("raw_observed_mean_err"),
        "normal_raw_mean_err": _mean("normal_raw_mean_err"),
        "normal_filtered_mean_err": _mean("normal_filtered_mean_err"),
        "normal_err_delta": _mean("normal_err_delta"),
        "normal_harm_rate": _mean("normal_harm_rate"),
        "moved_normal_mean_err": _mean("moved_normal_mean_err"),
        "corrupted_gain": _mean("corrupted_gain"),
    }
    # Lower is better.
    agg["objective"] = (
        float(agg["filtered_mean_err"])
        + 1.5 * max(float(agg["normal_err_delta"]), 0.0)
        + 1.0 * float(agg["normal_harm_rate"])
        - 0.8 * float(agg["corrupted_gain"])
    )
    return agg


def evaluate_config(
    cfg: FilterConfig,
    gt_library: dict[str, np.ndarray],
    pattern_names: list[str],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    case_metrics: list[dict[str, float | int]] = []
    case_artifacts: list[dict[str, Any]] = []

    for traj_i, (traj_name, gt) in enumerate(gt_library.items()):
        for pat_i, pattern in enumerate(pattern_names):
            combo_seed = int(seed + traj_i * 100 + pat_i * 7)
            rng = np.random.default_rng(combo_seed)
            normal = add_jitter(gt, rng=rng, sigma=0.7)
            raw, pat_info = inject_pattern(normal, pattern=pattern, rng=rng)
            out = run_filter_pipeline(raw, cfg)
            metrics = evaluate_one_case(gt, raw, pat_info, out)
            case_metrics.append(metrics)
            case_artifacts.append(
                {
                    "traj_name": traj_name,
                    "pattern": pattern,
                    "seed": combo_seed,
                    "gt": gt,
                    "raw": raw,
                    "filtered": out["filtered"],
                    "flagged_mask": out["flagged_mask"],
                    "normal_mask": (~np.asarray(pat_info["missing_mask"], dtype=bool))
                    & (~np.asarray(pat_info["injected_outlier_mask"], dtype=bool))
                    & out["observed_mask"],
                    "case_metrics": metrics,
                }
            )

    agg = aggregate_metrics(case_metrics)
    rows = []
    for art in case_artifacts:
        row = {
            "trajectory": art["traj_name"],
            "pattern": art["pattern"],
        }
        row.update(art["case_metrics"])
        rows.append(row)
    return agg, rows, case_artifacts


def _candidate_configs() -> list[FilterConfig]:
    coarse = []
    for macro_ratio, macro_sigma, micro_hw, micro_sigma, max_run, sg_w, keep in product(
        [0.04, 0.06, 0.08],
        [1.5, 2.0, 2.5],
        [1, 2, 3],
        [1.5, 2.0, 2.5],
        [1, 2],
        [5, 7],
        [0.0, 0.35, 0.6],
    ):
        coarse.append(
            FilterConfig(
                macro_ratio=macro_ratio,
                macro_sigma=macro_sigma,
                micro_hw=micro_hw,
                micro_sigma=micro_sigma,
                max_outlier_run=max_run,
                sg_window=sg_w,
                sg_polyorder=2,
                anchor_keep_ratio=keep,
            )
        )
    return coarse


def _refine_around(best: FilterConfig) -> list[FilterConfig]:
    macro_ratios = sorted({max(0.02, round(best.macro_ratio + d, 3)) for d in (-0.02, 0.0, 0.02)})
    macro_sigmas = sorted({max(1.0, round(best.macro_sigma + d, 2)) for d in (-0.5, 0.0, 0.5)})
    micro_hws = sorted({max(1, best.micro_hw + d) for d in (-1, 0, 1)})
    micro_sigmas = sorted({max(1.0, round(best.micro_sigma + d, 2)) for d in (-0.5, 0.0, 0.5)})
    max_runs = sorted({max(1, best.max_outlier_run + d) for d in (-1, 0, 1)})
    windows = sorted({w for w in (best.sg_window - 2, best.sg_window, best.sg_window + 2) if w >= 5 and w % 2 == 1})
    keeps = sorted({float(np.clip(round(best.anchor_keep_ratio + d, 2), 0.0, 0.8)) for d in (-0.2, 0.0, 0.2)})

    refined = []
    for macro_ratio, macro_sigma, micro_hw, micro_sigma, max_run, sg_w, keep in product(
        macro_ratios,
        macro_sigmas,
        micro_hws,
        micro_sigmas,
        max_runs,
        windows,
        keeps,
    ):
        refined.append(
            FilterConfig(
                macro_ratio=macro_ratio,
                macro_sigma=macro_sigma,
                micro_hw=micro_hw,
                micro_sigma=micro_sigma,
                max_outlier_run=max_run,
                sg_window=sg_w,
                sg_polyorder=2,
                anchor_keep_ratio=keep,
            )
        )
    return refined


def search_best_config(
    gt_library: dict[str, np.ndarray],
    pattern_names: list[str],
    seed: int,
) -> tuple[FilterConfig, dict[str, float], list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []

    best_cfg: FilterConfig | None = None
    best_agg: dict[str, float] | None = None

    for i, cfg in enumerate(_candidate_configs()):
        agg, _, _ = evaluate_config(cfg, gt_library, pattern_names, seed=seed)
        leaderboard.append({"phase": "coarse", "rank_seed": i, "config": asdict(cfg), **agg})
        if best_agg is None or float(agg["objective"]) < float(best_agg["objective"]):
            best_agg = agg
            best_cfg = cfg

    assert best_cfg is not None
    assert best_agg is not None

    # Iteration 2: refine around the current best (continuous auto experiment loop).
    for i, cfg in enumerate(_refine_around(best_cfg)):
        agg, _, _ = evaluate_config(cfg, gt_library, pattern_names, seed=seed)
        leaderboard.append({"phase": "refine", "rank_seed": i, "config": asdict(cfg), **agg})
        if float(agg["objective"]) < float(best_agg["objective"]):
            best_agg = agg
            best_cfg = cfg

    leaderboard_sorted = sorted(leaderboard, key=lambda r: float(r["objective"]))
    return best_cfg, best_agg, leaderboard_sorted


def plot_case(
    output_path: Path,
    title: str,
    gt: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    flagged_mask: np.ndarray,
    normal_mask: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=130)
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.8, label="GT")

    if normal_mask.any():
        ax.scatter(raw[normal_mask, 0], raw[normal_mask, 1], s=16, c="#1f77b4", alpha=0.9, label="Raw Normal")

    if flagged_mask.any():
        ax.scatter(filtered[flagged_mask, 0], filtered[flagged_mask, 1], s=42, marker="x", c="#d62728", lw=1.3, label="Flagged/Interpolated")

    ax.plot(filtered[:, 0], filtered[:, 1], c="#2ca02c", lw=2.0, label="Filtered")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_experiment(output_dir: Path, seed: int, force_config: FilterConfig | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_library = build_gt_library()
    pattern_names = [
        "P1_swiss_cheese",
        "P2_long_occlusion",
        "P3_isolated_spikes",
        "P4_consecutive_spikes",
        "P5_clean_only_jitter",
    ]

    if force_config is None:
        best_cfg, best_agg, leaderboard = search_best_config(gt_library, pattern_names, seed=seed)
    else:
        best_cfg = force_config
        best_agg, _, _ = evaluate_config(best_cfg, gt_library, pattern_names, seed=seed)
        leaderboard = []

    agg, rows, artifacts = evaluate_config(best_cfg, gt_library, pattern_names, seed=seed)

    plots_dir = output_dir / "plots"
    for art in artifacts:
        traj_name = str(art["traj_name"])
        pattern = str(art["pattern"])
        file_name = f"{traj_name}__{pattern}.png"
        case_metrics = art["case_metrics"]
        title = (
            f"{traj_name} | {pattern} | "
            f"Err={float(case_metrics['filtered_mean_err']):.3f} | "
            f"NormalΔ={float(case_metrics['normal_err_delta']):+.3f}"
        )
        plot_case(
            plots_dir / file_name,
            title,
            np.asarray(art["gt"]),
            np.asarray(art["raw"]),
            np.asarray(art["filtered"]),
            np.asarray(art["flagged_mask"]),
            np.asarray(art["normal_mask"]),
        )

    leaderboard_top20 = leaderboard[:20]
    summary = {
        "seed": int(seed),
        "best_config": asdict(best_cfg),
        "search_best_aggregate": best_agg,
        "final_aggregate": agg,
        "leaderboard_top20": leaderboard_top20,
        "case_metrics": rows,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also emit a compact CSV-like text for quick inspection in terminal.
    lines = [
        "trajectory,pattern,filtered_mean_err,normal_err_delta,normal_harm_rate,corrupted_gain",
    ]
    for r in rows:
        lines.append(
            f"{r['trajectory']},{r['pattern']},{float(r['filtered_mean_err']):.4f},"
            f"{float(r['normal_err_delta']):+.4f},{float(r['normal_harm_rate']):.4f},"
            f"{float(r['corrupted_gain']):+.4f}"
        )
    (output_dir / "case_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trajectory filter simulation + stress test + auto tuning")
    ap.add_argument("--output-dir", default="results/trajectory_filter_simulation", help="Directory for summary and plots")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--macro-ratio", type=float, default=None)
    ap.add_argument("--macro-sigma", type=float, default=None)
    ap.add_argument("--micro-hw", type=int, default=None)
    ap.add_argument("--micro-sigma", type=float, default=None)
    ap.add_argument("--max-outlier-run", type=int, default=None)
    ap.add_argument("--sg-window", type=int, default=None)
    ap.add_argument("--sg-polyorder", type=int, default=None)
    ap.add_argument("--anchor-keep-ratio", type=float, default=None)
    return ap.parse_args()


def _build_force_config(args: argparse.Namespace) -> FilterConfig | None:
    values = {
        "macro_ratio": args.macro_ratio,
        "macro_sigma": args.macro_sigma,
        "micro_hw": args.micro_hw,
        "micro_sigma": args.micro_sigma,
        "max_outlier_run": args.max_outlier_run,
        "sg_window": args.sg_window,
        "sg_polyorder": args.sg_polyorder,
        "anchor_keep_ratio": args.anchor_keep_ratio,
    }
    if all(v is None for v in values.values()):
        return None

    base = asdict(FilterConfig())
    for k, v in values.items():
        if v is not None:
            base[k] = v
    cfg = FilterConfig(**base)
    if cfg.sg_window % 2 == 0 or cfg.sg_window < 3:
        raise ValueError("sg_window must be odd and >= 3")
    if cfg.sg_polyorder >= cfg.sg_window:
        raise ValueError("sg_polyorder must be less than sg_window")
    return cfg


def main() -> None:
    args = parse_args()
    force_cfg = _build_force_config(args)
    summary = run_experiment(Path(args.output_dir), seed=int(args.seed), force_config=force_cfg)

    agg = summary["final_aggregate"]
    print("=== Trajectory Filter Simulation Completed ===")
    print("Best config:", json.dumps(summary["best_config"], ensure_ascii=False))
    print(
        "Aggregate: "
        f"filtered_mean_err={float(agg['filtered_mean_err']):.4f}, "
        f"normal_err_delta={float(agg['normal_err_delta']):+.4f}, "
        f"normal_harm_rate={float(agg['normal_harm_rate']):.4f}, "
        f"corrupted_gain={float(agg['corrupted_gain']):+.4f}, "
        f"objective={float(agg['objective']):.4f}"
    )


if __name__ == "__main__":
    main()
