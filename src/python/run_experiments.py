"""Run validation, paired Monte Carlo testing, ablations, and statistics."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.python.controllers import (
    AdaptiveADMMPID,
    DirectPolicyController,
    FixedPID,
    PolicyPIDController,
    SLLQG,
)
from src.python.model import NoPumpController, metrics, sample_day, sample_subject, simulate
from src.python.ppo_numpy import GaussianMLPPolicy


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POLICIES = RESULTS / "policies"
TEST_COUNT = 30
SUBJECT_SEED_BASE = 900_000
DAY_SEED_BASE = 1_000_000
PROPOSED = "RSP-ADMM-PID (proposed)"


METHOD_FILE_KEYS = {
    "Normal physiology": "normal",
    "Fixed PID": "fixed_pid",
    "SL-LQG": "sl_lqg",
    "Direct PPO": "direct_ppo",
    "PPO-PID": "ppo_pid",
    "ADMM-PID": "admm_pid",
    PROPOSED: "rsp_admm_pid",
}


def _case_seeds(index: int) -> tuple[int, int]:
    return SUBJECT_SEED_BASE + 17 * index, DAY_SEED_BASE + 31 * index


def select_policy(mode: str, wrapper, count: int = 8) -> tuple[GaussianMLPPolicy, dict]:
    candidates = []
    for seed in (11, 29, 47):
        path = POLICIES / f"{mode}_ppo_seed{seed}.npz"
        policy = GaussianMLPPolicy.load(path)
        rows = []
        for cohort in ("t1d", "t2d"):
            for index in range(count):
                subject = sample_subject(cohort, 500_000 + 17 * index)
                day = sample_day(600_000 + 31 * index)
                row = metrics(simulate(subject, day, wrapper(policy)))
                rows.append(row)
        score = float(
            np.mean(
                [
                    row["rmse_110"]
                    + 0.30 * row["tar_180"]
                    + 2.0 * row["tbr_70"]
                    for row in rows
                ]
            )
        )
        candidates.append(
            {
                "seed": seed,
                "path": str(path),
                "score": score,
                "tir": float(np.mean([r["tir_70_180"] for r in rows])),
                "rmse": float(np.mean([r["rmse_110"] for r in rows])),
                "policy": policy,
            }
        )
    selected = min(candidates, key=lambda item: item["score"])
    destination = POLICIES / f"{mode}_ppo_selected.npz"
    shutil.copy2(selected["path"], destination)
    public = [{k: v for k, v in item.items() if k != "policy"} for item in candidates]
    return selected["policy"], {"mode": mode, "selected_seed": selected["seed"], "candidates": public}


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "tir_70_180",
        "tight_80_140",
        "tbr_70",
        "tbr_54",
        "tar_180",
        "mean_glucose",
        "sd_glucose",
        "cv_glucose",
        "rmse_110",
        "normality_rmse",
        "total_insulin_u",
        "safety_interventions",
        "runtime_s",
    ]
    rows = []
    for (cohort, method), frame in raw.groupby(["cohort", "method"], sort=False):
        row = {"cohort": cohort, "method": method, "n": len(frame)}
        for metric in metric_columns:
            values = frame[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95"] = np.nan
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_ci95"] = (
                    1.96 * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = ["Fixed PID", "SL-LQG", "Direct PPO", "PPO-PID", "ADMM-PID"]
    for cohort in ("t1d", "t2d"):
        for metric, direction in (("normality_rmse", "less"), ("tir_70_180", "greater")):
            p_group = []
            group_indices = []
            proposed = (
                raw[(raw.cohort == cohort) & (raw.method == PROPOSED)]
                .sort_values("seed_index")
                .set_index("seed_index")[metric]
            )
            for baseline in baselines:
                comparator = (
                    raw[(raw.cohort == cohort) & (raw.method == baseline)]
                    .sort_values("seed_index")
                    .set_index("seed_index")[metric]
                )
                paired = pd.concat([proposed.rename("proposed"), comparator.rename("baseline")], axis=1).dropna()
                if np.allclose(paired.proposed, paired.baseline):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(
                        paired.proposed,
                        paired.baseline,
                        alternative=direction,
                        zero_method="wilcox",
                        method="auto",
                    )
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                difference = paired.proposed.to_numpy() - paired.baseline.to_numpy()
                rng = np.random.default_rng(77)
                bootstrap = np.array(
                    [
                        np.mean(rng.choice(difference, size=len(difference), replace=True))
                        for _ in range(5_000)
                    ]
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "metric": metric,
                        "alternative": direction,
                        "baseline": baseline,
                        "n": len(paired),
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                        "mean_paired_difference": float(np.mean(difference)),
                        "median_paired_difference": float(np.median(difference)),
                        "difference_ci95_low": float(np.percentile(bootstrap, 2.5)),
                        "difference_ci95_high": float(np.percentile(bootstrap, 97.5)),
                    }
                )
                p_group.append(p_value)
                group_indices.append(len(rows) - 1)
            adjusted = _holm_adjust(np.asarray(p_group))
            for index, p_adjusted in zip(group_indices, adjusted):
                rows[index]["p_holm"] = float(p_adjusted)
    return pd.DataFrame(rows)


def run_main_experiment(direct_policy, pid_policy) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    normal_references = {}
    trajectories: dict[str, np.ndarray] = {}
    raw_rows = []

    # Matched healthy references use the same meal/exercise realization.
    for index in range(TEST_COUNT):
        subject_seed, day_seed = _case_seeds(index)
        normal_subject = sample_subject("normal", subject_seed)
        day = sample_day(day_seed)
        start = time.perf_counter()
        trajectory = simulate(normal_subject, day, NoPumpController())
        elapsed = time.perf_counter() - start
        normal_references[index] = trajectory.glucose_true.copy()
        row = metrics(trajectory, trajectory.glucose_true)
        row.update(
            {
                "cohort": "normal",
                "method": "Normal physiology",
                "seed_index": index,
                "subject_seed": subject_seed,
                "day_seed": day_seed,
                "runtime_s": elapsed,
            }
        )
        raw_rows.append(row)
        trajectories[f"normal_normal_glucose_{index:02d}"] = trajectory.glucose_true
        trajectories[f"normal_normal_rate_{index:02d}"] = trajectory.insulin_rate

    factories = {
        "Fixed PID": lambda: FixedPID(),
        "SL-LQG": lambda: SLLQG(),
        "Direct PPO": lambda: DirectPolicyController(direct_policy),
        "PPO-PID": lambda: PolicyPIDController(pid_policy),
        "ADMM-PID": lambda: AdaptiveADMMPID(robust=False, safety=False),
        PROPOSED: lambda: AdaptiveADMMPID(robust=True, safety=True),
    }
    for cohort in ("t1d", "t2d"):
        for method, factory in factories.items():
            print(f"Evaluating {cohort} / {method}", flush=True)
            for index in range(TEST_COUNT):
                subject_seed, day_seed = _case_seeds(index)
                subject = sample_subject(cohort, subject_seed)
                day = sample_day(day_seed)
                start = time.perf_counter()
                trajectory = simulate(subject, day, factory())
                elapsed = time.perf_counter() - start
                row = metrics(trajectory, normal_references[index])
                row.update(
                    {
                        "cohort": cohort,
                        "method": method,
                        "seed_index": index,
                        "subject_seed": subject_seed,
                        "day_seed": day_seed,
                        "runtime_s": elapsed,
                    }
                )
                raw_rows.append(row)
                prefix = f"{cohort}_{METHOD_FILE_KEYS[method]}_{index:02d}"
                trajectories[f"{prefix}_glucose"] = trajectory.glucose_true
                trajectories[f"{prefix}_cgm"] = trajectory.glucose_cgm
                trajectories[f"{prefix}_rate"] = trajectory.insulin_rate
                trajectories[f"{prefix}_gains"] = trajectory.gains
                trajectories[f"{prefix}_safety"] = trajectory.safety_interventions.astype(np.uint8)
    trajectories["time_min"] = np.arange(0.0, 24.0 * 60.0 + 5.0, 5.0)
    return pd.DataFrame(raw_rows), trajectories


def run_ablation(direct_raw: pd.DataFrame, count: int = 15) -> pd.DataFrame:
    rows = []
    variants = {
        "ADMM-PID": None,
        "Scenario ADMM-PID": lambda: AdaptiveADMMPID(robust=True, safety=False),
        "Safety-projected ADMM-PID": lambda: AdaptiveADMMPID(robust=False, safety=True),
        PROPOSED: None,
    }
    for cohort in ("t1d", "t2d"):
        for variant, factory in variants.items():
            if factory is None:
                reused = direct_raw[
                    (direct_raw.cohort == cohort)
                    & (direct_raw.method == variant)
                    & (direct_raw.seed_index < count)
                ].copy().drop(columns=["method"])
                reused["variant"] = variant
                rows.append(reused)
                continue
            print(f"Ablation {cohort} / {variant}", flush=True)
            variant_rows = []
            for index in range(count):
                subject_seed, day_seed = _case_seeds(index)
                subject = sample_subject(cohort, subject_seed)
                day = sample_day(day_seed)
                trajectory = simulate(subject, day, factory())
                row = metrics(trajectory)
                row.update({"cohort": cohort, "variant": variant, "seed_index": index})
                variant_rows.append(row)
            rows.append(pd.DataFrame(variant_rows))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    direct_policy, direct_selection = select_policy("direct", DirectPolicyController)
    pid_policy, pid_selection = select_policy("pid", PolicyPIDController)
    with (RESULTS / "policy_selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"direct": direct_selection, "pid": pid_selection}, handle, indent=2)
    print(
        f"Selected direct seed {direct_selection['selected_seed']} and "
        f"PID seed {pid_selection['selected_seed']}",
        flush=True,
    )

    raw, trajectories = run_main_experiment(direct_policy, pid_policy)
    raw.to_csv(RESULTS / "per_subject_metrics.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(RESULTS / "summary_mean_std.csv", index=False)
    stats = paired_statistics(raw)
    stats.to_csv(RESULTS / "paired_wilcoxon_holm.csv", index=False)
    np.savez_compressed(RESULTS / "trajectories.npz", **trajectories)

    ablation = run_ablation(raw)
    ablation.to_csv(RESULTS / "ablation_per_subject.csv", index=False)
    ablation_summary = summarize(
        ablation.rename(columns={"variant": "method"}).assign(runtime_s=np.nan)
    )
    ablation_summary.to_csv(RESULTS / "ablation_summary.csv", index=False)

    manifest = {
        "test_subjects_per_cohort": TEST_COUNT,
        "ablation_subjects_per_cohort": 15,
        "subject_seed_base": SUBJECT_SEED_BASE,
        "day_seed_base": DAY_SEED_BASE,
        "control_period_min": 5,
        "integration_period_min": 1,
        "primary_method": PROPOSED,
        "selected_direct_seed": direct_selection["selected_seed"],
        "selected_pid_seed": pid_selection["selected_seed"],
    }
    with (RESULTS / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Experiment complete.", flush=True)


if __name__ == "__main__":
    main()
