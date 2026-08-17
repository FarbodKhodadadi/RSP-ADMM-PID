"""Extended Bergman virtual-cohort benchmark used in the ICBME paper.

The implementation is deliberately dependency-light (NumPy only) so that the
same equations can be mirrored in MATLAB.  Time is measured in minutes,
glucose in mg/dL, plasma insulin in mU/L, and pump commands in U/h.

This is an in-silico control benchmark, not a dosing tool.  It must never be
used to make clinical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


COHORTS = ("normal", "t1d", "t2d")


@dataclass(frozen=True)
class Subject:
    """One virtual subject and its time-invariant physiological parameters."""

    cohort: str
    gb: float
    ib: float
    p1: float
    p2: float
    p3: float
    n: float
    secretion_gain: float
    basal_rate: float
    max_rate: float
    insulin_gain: float
    tau_iob: float
    tau_cgm: float
    cgm_bias: float
    cgm_noise_sd: float
    glucose_volume_dl: float
    exercise_gain: float


@dataclass(frozen=True)
class DayScenario:
    """Exogenous events and matched noise for a 24-hour experiment."""

    meal_times: np.ndarray
    meal_carbs_g: np.ndarray
    meal_taus: np.ndarray
    exercise_start: float
    exercise_duration: float
    exercise_intensity: float
    cgm_noise: np.ndarray
    process_noise: np.ndarray
    initial_z: np.ndarray


@dataclass
class Trajectory:
    """Simulation output sampled at the controller period."""

    time_min: np.ndarray
    state: np.ndarray
    glucose_true: np.ndarray
    glucose_cgm: np.ndarray
    insulin_rate: np.ndarray
    gains: np.ndarray
    reward: np.ndarray
    safety_interventions: np.ndarray


def _lognormal_multiplier(rng: np.random.Generator, cv: float) -> float:
    sigma = np.sqrt(np.log1p(cv * cv))
    return float(np.exp(sigma * rng.normal() - 0.5 * sigma * sigma))


def nominal_subject(cohort: str) -> Subject:
    """Return the nominal phenotype used by model-based controllers.

    The three phenotypes differ through glucose effectiveness, insulin
    sensitivity, beta-cell responsivity, and background basal therapy.  The
    normal phenotype is a no-pump physiological reference.
    """

    cohort = cohort.lower()
    if cohort == "normal":
        return Subject(
            cohort=cohort,
            gb=100.0,
            ib=15.0,
            p1=0.028,
            p2=0.025,
            p3=1.30e-5,
            n=5.0 / 54.0,
            secretion_gain=0.060,
            basal_rate=0.0,
            max_rate=0.0,
            insulin_gain=0.90,
            tau_iob=60.0,
            tau_cgm=10.0,
            cgm_bias=0.0,
            cgm_noise_sd=4.0,
            glucose_volume_dl=130.0,
            exercise_gain=0.60,
        )
    if cohort == "t1d":
        return Subject(
            cohort=cohort,
            gb=110.0,
            ib=15.0,
            p1=0.012,
            p2=0.025,
            p3=1.30e-5,
            n=5.0 / 54.0,
            secretion_gain=0.0,
            basal_rate=0.90,
            max_rate=5.0,
            insulin_gain=0.90,
            tau_iob=60.0,
            tau_cgm=10.0,
            cgm_bias=0.0,
            cgm_noise_sd=5.0,
            glucose_volume_dl=130.0,
            exercise_gain=0.75,
        )
    if cohort == "t2d":
        return Subject(
            cohort=cohort,
            gb=110.0,
            ib=20.0,
            p1=0.018,
            p2=0.025,
            p3=0.62 * 1.30e-5,
            n=5.0 / 54.0,
            secretion_gain=0.020,
            basal_rate=0.45,
            max_rate=4.0,
            insulin_gain=0.85,
            tau_iob=70.0,
            tau_cgm=10.0,
            cgm_bias=0.0,
            cgm_noise_sd=5.0,
            glucose_volume_dl=135.0,
            exercise_gain=0.55,
        )
    raise ValueError(f"Unknown cohort: {cohort!r}")


def sample_subject(cohort: str, seed: int) -> Subject:
    """Sample a virtual subject from reproducible lognormal uncertainty.

    Positive parameters use lognormal multipliers; sensor bias is normal.  The
    ranges are domain-randomization choices for robustness testing rather than
    claims about a clinically validated population distribution.
    """

    rng = np.random.default_rng(int(seed))
    s = nominal_subject(cohort)
    body_mass_multiplier = float(np.clip(_lognormal_multiplier(rng, 0.12), 0.72, 1.35))
    return replace(
        s,
        gb=float(s.gb + rng.normal(0.0, 3.0)),
        ib=float(max(5.0, s.ib * _lognormal_multiplier(rng, 0.12))),
        p1=float(s.p1 * _lognormal_multiplier(rng, 0.22)),
        p2=float(s.p2 * _lognormal_multiplier(rng, 0.18)),
        p3=float(s.p3 * _lognormal_multiplier(rng, 0.28)),
        n=float(s.n * _lognormal_multiplier(rng, 0.15)),
        secretion_gain=float(s.secretion_gain * _lognormal_multiplier(rng, 0.30))
        if s.secretion_gain > 0
        else 0.0,
        basal_rate=float(s.basal_rate * _lognormal_multiplier(rng, 0.12))
        if s.basal_rate > 0
        else 0.0,
        insulin_gain=float(s.insulin_gain * _lognormal_multiplier(rng, 0.18)),
        tau_iob=float(s.tau_iob * _lognormal_multiplier(rng, 0.18)),
        tau_cgm=float(np.clip(s.tau_cgm * _lognormal_multiplier(rng, 0.20), 5.0, 18.0)),
        cgm_bias=float(rng.normal(0.0, 3.0)),
        cgm_noise_sd=float(np.clip(s.cgm_noise_sd * _lognormal_multiplier(rng, 0.15), 2.5, 8.0)),
        glucose_volume_dl=float(s.glucose_volume_dl * body_mass_multiplier),
        exercise_gain=float(s.exercise_gain * _lognormal_multiplier(rng, 0.25)),
    )


def _truncated_normal(
    rng: np.random.Generator, mean: float, sd: float, low: float, high: float
) -> float:
    for _ in range(100):
        x = rng.normal(mean, sd)
        if low <= x <= high:
            return float(x)
    return float(np.clip(x, low, high))


def sample_day(seed: int, n_steps: int = 289) -> DayScenario:
    """Sample three main meals, an optional snack, exercise, and matched noise."""

    rng = np.random.default_rng(int(seed))
    means = np.array([8.0, 13.0, 19.0]) * 60.0
    sds = np.array([25.0, 35.0, 40.0])
    lows = np.array([6.5, 11.0, 17.0]) * 60.0
    highs = np.array([10.0, 15.5, 21.5]) * 60.0
    meal_times = np.array(
        [_truncated_normal(rng, m, sd, lo, hi) for m, sd, lo, hi in zip(means, sds, lows, highs)]
    )
    carb_medians = np.array([48.0, 68.0, 74.0])
    meal_carbs = np.array(
        [np.clip(m * _lognormal_multiplier(rng, 0.25), 20.0, 120.0) for m in carb_medians]
    )
    meal_taus = np.array(
        [np.clip(45.0 * _lognormal_multiplier(rng, 0.22), 25.0, 80.0) for _ in range(3)]
    )
    if rng.random() < 0.60:
        meal_times = np.append(
            meal_times, _truncated_normal(rng, 22.0 * 60.0, 25.0, 20.5 * 60.0, 23.5 * 60.0)
        )
        meal_carbs = np.append(meal_carbs, np.clip(24.0 * _lognormal_multiplier(rng, 0.30), 10.0, 45.0))
        meal_taus = np.append(meal_taus, np.clip(38.0 * _lognormal_multiplier(rng, 0.20), 22.0, 65.0))
    order = np.argsort(meal_times)
    meal_times = meal_times[order]
    meal_carbs = meal_carbs[order]
    meal_taus = meal_taus[order]

    if rng.random() < 0.78:
        ex_start = _truncated_normal(rng, 17.2 * 60.0, 50.0, 14.5 * 60.0, 20.5 * 60.0)
        ex_duration = float(rng.uniform(30.0, 75.0))
        ex_intensity = float(rng.uniform(0.35, 0.85))
    else:
        ex_start, ex_duration, ex_intensity = np.inf, 0.0, 0.0

    return DayScenario(
        meal_times=meal_times,
        meal_carbs_g=meal_carbs,
        meal_taus=meal_taus,
        exercise_start=ex_start,
        exercise_duration=ex_duration,
        exercise_intensity=ex_intensity,
        cgm_noise=rng.normal(0.0, 1.0, int(n_steps)),
        process_noise=rng.normal(0.0, 0.70, int(n_steps)),
        initial_z=rng.normal(0.0, 1.0, 2),
    )


def meal_appearance(t_min: float, subject: Subject, scenario: DayScenario) -> float:
    """Two-compartment gamma meal appearance in mg/dL/min."""

    lag = float(t_min) - scenario.meal_times
    active = lag >= 0.0
    if not np.any(active):
        return 0.0
    lag = lag[active]
    tau = scenario.meal_taus[active]
    absorbed_mg_dl = 0.90 * 1000.0 * scenario.meal_carbs_g[active] / subject.glucose_volume_dl
    kernel = lag * np.exp(-lag / tau) / (tau * tau)
    return float(np.sum(absorbed_mg_dl * kernel))


def exercise_multiplier(t_min: float, subject: Subject, scenario: DayScenario) -> float:
    """Smooth exercise sensitivity increase with a decaying post-exercise tail."""

    if not np.isfinite(scenario.exercise_start) or t_min < scenario.exercise_start:
        return 1.0
    elapsed = float(t_min - scenario.exercise_start)
    rise = 1.0 - np.exp(-elapsed / 10.0)
    if elapsed <= scenario.exercise_duration:
        activity = rise
    else:
        end_level = 1.0 - np.exp(-scenario.exercise_duration / 10.0)
        activity = end_level * np.exp(-(elapsed - scenario.exercise_duration) / 100.0)
    return 1.0 + subject.exercise_gain * scenario.exercise_intensity * activity


def circadian_sensitivity(t_min: float) -> float:
    """Bounded daily modulation of insulin sensitivity."""

    return float(1.0 + 0.12 * np.sin(2.0 * np.pi * (t_min - 15.0 * 60.0) / (24.0 * 60.0)))


def rhs(
    t_min: float,
    state: np.ndarray,
    insulin_rate: float,
    subject: Subject,
    scenario: DayScenario | None,
    *,
    appearance_override: float | None = None,
    sensitivity_scale: float = 1.0,
) -> np.ndarray:
    """Continuous-time extended Bergman dynamics.

    State order is ``[G, X, I, IOB, G_cgm_lag]``.  ``basal_rate`` is the
    subject's pre-programmed operating-point therapy, so the plasma-insulin
    equation is driven by deviations from that rate.
    """

    g, x, insulin, iob, g_sensor = np.asarray(state, dtype=float)
    u = float(np.clip(insulin_rate, 0.0, subject.max_rate)) if subject.max_rate > 0 else 0.0
    if appearance_override is not None:
        ra = float(appearance_override)
    elif scenario is None:
        ra = 0.0
    else:
        ra = meal_appearance(t_min, subject, scenario)
    ex = 1.0 if scenario is None else exercise_multiplier(t_min, subject, scenario)
    alpha = sensitivity_scale * ex * circadian_sensitivity(t_min)
    secretion = subject.secretion_gain * max(g - subject.gb, 0.0)

    dg = -subject.p1 * (g - subject.gb) - alpha * x * g + ra
    dx = -subject.p2 * x + subject.p3 * (insulin - subject.ib)
    di = -subject.n * (insulin - subject.ib) + subject.insulin_gain * (u - subject.basal_rate) + secretion
    diob = u / 60.0 - iob / subject.tau_iob
    dgs = (g - g_sensor) / subject.tau_cgm
    return np.array([dg, dx, di, diob, dgs], dtype=float)


def rk4_step(
    t_min: float,
    state: np.ndarray,
    insulin_rate: float,
    subject: Subject,
    scenario: DayScenario | None,
    dt_min: float,
    *,
    appearance_override: float | None = None,
    sensitivity_scale: float = 1.0,
) -> np.ndarray:
    """One fourth-order Runge-Kutta step with conservative state clipping."""

    f = lambda tt, xx: rhs(
        tt,
        xx,
        insulin_rate,
        subject,
        scenario,
        appearance_override=appearance_override,
        sensitivity_scale=sensitivity_scale,
    )
    x0 = np.asarray(state, dtype=float)
    k1 = f(t_min, x0)
    k2 = f(t_min + 0.5 * dt_min, x0 + 0.5 * dt_min * k1)
    k3 = f(t_min + 0.5 * dt_min, x0 + 0.5 * dt_min * k2)
    k4 = f(t_min + dt_min, x0 + dt_min * k3)
    out = x0 + dt_min * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    out[0] = np.clip(out[0], 35.0, 500.0)
    out[1] = np.clip(out[1], -0.03, 0.08)
    out[2] = np.clip(out[2], 0.0, 600.0)
    out[3] = np.clip(out[3], 0.0, 30.0)
    out[4] = np.clip(out[4], 35.0, 500.0)
    return out


def advance_interval(
    t_min: float,
    state: np.ndarray,
    insulin_rate: float,
    subject: Subject,
    scenario: DayScenario | None,
    control_dt: float = 5.0,
    integration_dt: float = 1.0,
    *,
    appearance_override: float | None = None,
    sensitivity_scale: float = 1.0,
) -> np.ndarray:
    """Advance one zero-order-held controller interval."""

    n = int(round(control_dt / integration_dt))
    if not np.isclose(n * integration_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of integration_dt")
    out = np.asarray(state, dtype=float).copy()
    for j in range(n):
        out = rk4_step(
            t_min + j * integration_dt,
            out,
            insulin_rate,
            subject,
            scenario,
            integration_dt,
            appearance_override=appearance_override,
            sensitivity_scale=sensitivity_scale,
        )
    return out


def initial_state(subject: Subject, scenario: DayScenario) -> np.ndarray:
    g0 = subject.gb + 4.0 * scenario.initial_z[0]
    i0 = max(0.0, subject.ib + scenario.initial_z[1])
    iob0 = subject.basal_rate * subject.tau_iob / 60.0
    return np.array([g0, 0.0, i0, iob0, g0], dtype=float)


def clinical_reward(glucose: float, rate: float, subject: Subject, reference: float = 110.0) -> float:
    """Bounded per-step objective shared by RL training and evaluation."""

    g = float(glucose)
    track = ((g - reference) / 45.0) ** 2
    low = 8.0 * (max(70.0 - g, 0.0) / 20.0) ** 2
    very_low = 14.0 * (max(54.0 - g, 0.0) / 16.0) ** 2
    high = 1.6 * (max(g - 180.0, 0.0) / 60.0) ** 2
    effort = 0.015 * ((rate - subject.basal_rate) / max(subject.max_rate, 1.0)) ** 2
    return float(-(track + low + very_low + high + effort))


def simulate(
    subject: Subject,
    scenario: DayScenario,
    controller,
    *,
    horizon_min: float = 24.0 * 60.0,
    control_dt: float = 5.0,
    integration_dt: float = 1.0,
) -> Trajectory:
    """Run one deterministic paired simulation for a controller."""

    n_steps = int(round(horizon_min / control_dt)) + 1
    if len(scenario.cgm_noise) < n_steps or len(scenario.process_noise) < n_steps:
        raise ValueError("scenario noise arrays are shorter than the requested horizon")
    times = np.arange(n_steps, dtype=float) * control_dt
    states = np.zeros((n_steps, 5), dtype=float)
    g_cgm = np.zeros(n_steps, dtype=float)
    rates = np.zeros(n_steps, dtype=float)
    rewards = np.zeros(n_steps, dtype=float)
    gains = np.full((n_steps, 3), np.nan, dtype=float)
    interventions = np.zeros(n_steps, dtype=bool)
    states[0] = initial_state(subject, scenario)
    controller.reset(subject=subject, control_dt=control_dt)

    for k, t_min in enumerate(times[:-1]):
        measurement = states[k, 4] + subject.cgm_bias + subject.cgm_noise_sd * scenario.cgm_noise[k]
        g_cgm[k] = float(np.clip(measurement, 40.0, 400.0))
        rate, info = controller.act(float(t_min), g_cgm[k])
        rate = float(np.clip(rate, 0.0, subject.max_rate)) if subject.max_rate > 0 else 0.0
        rates[k] = rate
        if info is not None:
            if "gains" in info:
                gains[k] = np.asarray(info["gains"], dtype=float)
            interventions[k] = bool(info.get("safety_intervened", False))
        states[k + 1] = advance_interval(
            float(t_min), states[k], rate, subject, scenario, control_dt, integration_dt
        )
        states[k + 1, 0] = np.clip(states[k + 1, 0] + scenario.process_noise[k + 1], 35.0, 500.0)
        rewards[k] = clinical_reward(states[k, 0], rate, subject)

    g_cgm[-1] = float(
        np.clip(states[-1, 4] + subject.cgm_bias + subject.cgm_noise_sd * scenario.cgm_noise[-1], 40.0, 400.0)
    )
    rates[-1] = rates[-2]
    rewards[-1] = clinical_reward(states[-1, 0], rates[-1], subject)
    if np.all(np.isfinite(gains[-2])):
        gains[-1] = gains[-2]
    interventions[-1] = interventions[-2]
    return Trajectory(
        time_min=times,
        state=states,
        glucose_true=states[:, 0].copy(),
        glucose_cgm=g_cgm,
        insulin_rate=rates,
        gains=gains,
        reward=rewards,
        safety_interventions=interventions,
    )


def metrics(trajectory: Trajectory, normal_reference: np.ndarray | None = None) -> dict[str, float]:
    """Compute CGM consensus metrics and paper-specific tracking measures."""

    g = np.asarray(trajectory.glucose_true, dtype=float)
    u = np.asarray(trajectory.insulin_rate, dtype=float)
    dt_h = float(np.median(np.diff(trajectory.time_min))) / 60.0
    out = {
        "tir_70_180": 100.0 * float(np.mean((g >= 70.0) & (g <= 180.0))),
        "tight_80_140": 100.0 * float(np.mean((g >= 80.0) & (g <= 140.0))),
        "tbr_70": 100.0 * float(np.mean(g < 70.0)),
        "tbr_54": 100.0 * float(np.mean(g < 54.0)),
        "tar_180": 100.0 * float(np.mean(g > 180.0)),
        "mean_glucose": float(np.mean(g)),
        "sd_glucose": float(np.std(g, ddof=1)),
        "cv_glucose": 100.0 * float(np.std(g, ddof=1) / max(np.mean(g), 1e-9)),
        "rmse_110": float(np.sqrt(np.mean((g - 110.0) ** 2))),
        "total_insulin_u": float(np.sum(u[:-1]) * dt_h),
        "safety_interventions": float(np.sum(trajectory.safety_interventions[:-1])),
        "return": float(np.sum(trajectory.reward[:-1])),
    }
    if normal_reference is not None:
        ref = np.asarray(normal_reference, dtype=float)
        if ref.shape != g.shape:
            raise ValueError("normal_reference and trajectory must have the same shape")
        out["normality_rmse"] = float(np.sqrt(np.mean((g - ref) ** 2)))
    else:
        out["normality_rmse"] = np.nan
    return out


class NoPumpController:
    """Normal physiology reference: no exogenous insulin."""

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject

    def act(self, t_min: float, glucose_cgm: float):
        return 0.0, {"safety_intervened": False}


def cohort_seed_pairs(base_seed: int, count: int) -> Iterable[tuple[int, int]]:
    """Yield independent subject/day seeds with a stable public convention."""

    for index in range(int(count)):
        yield int(base_seed + 17 * index), int(base_seed + 100_000 + 31 * index)
