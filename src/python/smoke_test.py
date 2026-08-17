#!/usr/bin/env python3
"""Fast deterministic end-to-end checks for every published controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
POLICIES = ROOT / "results" / "policies"


def _assert_valid(label: str, trajectory) -> None:
    values = metrics(trajectory)
    assert len(trajectory.time_min) == 289, f"{label}: unexpected horizon"
    assert np.all(np.isfinite(trajectory.glucose_true)), f"{label}: non-finite glucose"
    assert np.all(np.isfinite(trajectory.insulin_rate)), f"{label}: non-finite insulin"
    assert 0.0 <= values["tir_70_180"] <= 100.0, f"{label}: invalid TIR"
    assert values["tbr_54"] <= values["tbr_70"] + 1e-12, f"{label}: invalid TBR"
    assert np.min(trajectory.glucose_true) >= 35.0, f"{label}: state clip violated"
    assert np.max(trajectory.glucose_true) <= 500.0, f"{label}: state clip violated"
    print(
        f"{label:28s} TIR={values['tir_70_180']:6.2f}% "
        f"TBR={values['tbr_70']:5.2f}% insulin={values['total_insulin_u']:6.2f} U"
    )


def main() -> None:
    day = sample_day(1_234_567)
    normal = sample_subject("normal", 765_432)
    _assert_valid("Normal physiology", simulate(normal, day, NoPumpController()))

    direct = GaussianMLPPolicy.load(POLICIES / "direct_ppo_selected.npz")
    pid = GaussianMLPPolicy.load(POLICIES / "pid_ppo_selected.npz")
    factories = {
        "Fixed PID": lambda: FixedPID(),
        "SL-LQG": lambda: SLLQG(),
        "Direct PPO": lambda: DirectPolicyController(direct),
        "PPO-PID": lambda: PolicyPIDController(pid),
        "ADMM-PID": lambda: AdaptiveADMMPID(robust=False, safety=False),
        "RSP-ADMM-PID": lambda: AdaptiveADMMPID(robust=True, safety=True),
    }
    for cohort in ("t1d", "t2d"):
        subject = sample_subject(cohort, 765_432)
        for method, factory in factories.items():
            _assert_valid(f"{cohort.upper()} / {method}", simulate(subject, day, factory()))
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()

