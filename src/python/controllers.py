"""Controllers for the virtual-cohort glucose regulation benchmark.

The module contains the classical, model-based, RL-policy wrappers, and the
proposed robust safety-projected ADMM-PID controller.  Every controller uses
the same CGM-only interface; IOB is reconstructed exactly from past pump
commands.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.python.model import Subject, advance_interval, nominal_subject


GAIN_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
GAIN_MAX = np.array([0.080, 0.00080, 0.250], dtype=float)
NOMINAL_GAINS = {
    "t1d": np.array([0.028, 0.00012, 0.045], dtype=float),
    "t2d": np.array([0.024, 0.00010, 0.040], dtype=float),
}


class MinimalEKF:
    """Small EKF for ``[G, X, I]`` plus exact command-derived IOB."""

    def __init__(self, model: Subject, control_dt: float):
        self.model = model
        self.dt = float(control_dt)
        self.initialized = False
        self.x = np.array([model.gb, 0.0, model.ib], dtype=float)
        self.p = np.diag([36.0, 2.5e-5, 25.0])
        self.q = np.diag([7.0, 2.0e-7, 2.0])
        self.r = max(model.cgm_noise_sd**2, 9.0)
        self.iob = model.basal_rate * model.tau_iob / 60.0
        self.u_prev = model.basal_rate

    def _derivative(self, x: np.ndarray, u: float) -> np.ndarray:
        g, remote, insulin = x
        secretion = self.model.secretion_gain * max(g - self.model.gb, 0.0)
        return np.array(
            [
                -self.model.p1 * (g - self.model.gb) - remote * g,
                -self.model.p2 * remote + self.model.p3 * (insulin - self.model.ib),
                -self.model.n * (insulin - self.model.ib)
                + self.model.insulin_gain * (u - self.model.basal_rate)
                + secretion,
            ],
            dtype=float,
        )

    def _rk4(self, x: np.ndarray, u: float) -> np.ndarray:
        dt = self.dt
        k1 = self._derivative(x, u)
        k2 = self._derivative(x + 0.5 * dt * k1, u)
        k3 = self._derivative(x + 0.5 * dt * k2, u)
        k4 = self._derivative(x + dt * k3, u)
        out = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        out[0] = np.clip(out[0], 35.0, 500.0)
        out[1] = np.clip(out[1], -0.03, 0.08)
        out[2] = np.clip(out[2], 0.0, 600.0)
        return out

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        g, remote, _ = x
        dsecretion = self.model.secretion_gain if g > self.model.gb else 0.0
        return np.array(
            [
                [-self.model.p1 - remote, -g, 0.0],
                [0.0, -self.model.p2, self.model.p3],
                [dsecretion, 0.0, -self.model.n],
            ],
            dtype=float,
        )

    def update(self, measurement: float) -> np.ndarray:
        y = float(measurement)
        if not self.initialized:
            self.x[0] = y
            self.initialized = True
            return self.full_state()

        f = np.eye(3) + self.dt * self._jacobian(self.x)
        self.x = self._rk4(self.x, self.u_prev)
        self.p = f @ self.p @ f.T + self.q
        h = np.array([[1.0, 0.0, 0.0]])
        innovation = y - float(h @ self.x)
        s = float(h @ self.p @ h.T + self.r)
        k = (self.p @ h.T / s).reshape(3)
        self.x = self.x + k * innovation
        self.p = (np.eye(3) - np.outer(k, h.reshape(3))) @ self.p
        self.x[0] = np.clip(self.x[0], 35.0, 500.0)
        self.x[1] = np.clip(self.x[1], -0.03, 0.08)
        self.x[2] = np.clip(self.x[2], 0.0, 600.0)
        decay = np.exp(-self.dt / self.model.tau_iob)
        self.iob = decay * self.iob + self.u_prev * self.model.tau_iob / 60.0 * (1.0 - decay)
        return self.full_state()

    def set_command(self, rate: float) -> None:
        self.u_prev = float(np.clip(rate, 0.0, self.model.max_rate))

    def full_state(self) -> np.ndarray:
        return np.array([self.x[0], self.x[1], self.x[2], self.iob, self.x[0]], dtype=float)


class FixedPID:
    """Filtered, anti-windup fixed-gain PID around a known basal rate."""

    name = "Fixed PID"

    def __init__(self, gains: np.ndarray | None = None, reference: float = 110.0):
        self.gains_input = None if gains is None else np.asarray(gains, dtype=float)
        self.reference = float(reference)

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject
        self.dt = float(control_dt)
        self.gains = (
            self.gains_input.copy()
            if self.gains_input is not None
            else NOMINAL_GAINS[subject.cohort].copy()
        )
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.first = True

    def act(self, t_min: float, glucose_cgm: float):
        error = float(glucose_cgm - self.reference)
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / self.dt
        alpha = 10.0 / (10.0 + self.dt)
        self.derivative = alpha * self.derivative + (1.0 - alpha) * raw_derivative
        candidate_integral = np.clip(self.integral + error * self.dt, -12_000.0, 12_000.0)
        features = np.array([error, candidate_integral, self.derivative])
        raw = self.subject.basal_rate + float(self.gains @ features)
        rate = float(np.clip(raw, 0.0, self.subject.max_rate))
        saturated_high = raw > self.subject.max_rate and error > 0.0
        saturated_low = raw < 0.0 and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = float(candidate_integral)
        self.prev_error = error
        self.first = False
        return rate, {"gains": self.gains.copy(), "safety_intervened": False}


class SLLQG:
    """Successive-linearization EKF-LQR baseline (SL-LQG)."""

    name = "SL-LQG"

    def __init__(self, reference: float = 110.0):
        self.reference = float(reference)

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject
        self.dt = float(control_dt)
        model = replace(
            nominal_subject(subject.cohort),
            basal_rate=subject.basal_rate,
            max_rate=subject.max_rate,
        )
        self.model = model
        self.observer = MinimalEKF(model, self.dt)
        self.last_k = np.zeros(3)

    def _linear_model(self, xhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g, remote, _ = xhat[:3]
        secretion_derivative = self.model.secretion_gain if g > self.model.gb else 0.0
        a = np.array(
            [
                [-self.model.p1 - remote, -g, 0.0],
                [0.0, -self.model.p2, self.model.p3],
                [secretion_derivative, 0.0, -self.model.n],
            ]
        )
        b = np.array([[0.0], [0.0], [self.model.insulin_gain]])
        block = np.zeros((4, 4))
        block[:3, :3] = a
        block[:3, 3:] = b
        discrete = expm(block * self.dt)
        return discrete[:3, :3], discrete[:3, 3:]

    def act(self, t_min: float, glucose_cgm: float):
        xhat = self.observer.update(glucose_cgm)
        ad, bd = self._linear_model(xhat)
        q = np.diag([1.0 / 35.0**2, 1.0 / 0.012**2, 1.0 / 70.0**2])
        r = np.array([[0.040]])
        try:
            p = solve_discrete_are(ad, bd, q, r)
            k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad).reshape(3)
            if np.all(np.isfinite(k)):
                self.last_k = k
        except Exception:
            k = self.last_k
        deviation = np.array([xhat[0] - self.reference, xhat[1], xhat[2] - self.model.ib])
        rate = self.subject.basal_rate - float(self.last_k @ deviation)
        rate = float(np.clip(rate, 0.0, self.subject.max_rate))
        self.observer.set_command(rate)
        return rate, {"safety_intervened": False}


class AdaptiveADMMPID:
    """Online ADMM gain tuner, optionally robust and safety projected.

    ``robust=False, safety=False`` is the ADMM-PID ablation.  The proposed
    RSP-ADMM-PID uses a small uncertainty scenario set and a one-dimensional
    predictive command projection.
    """

    def __init__(
        self,
        *,
        robust: bool,
        safety: bool,
        reference: float = 110.0,
        admm_iterations: int = 3,
        prediction_steps: int = 8,
        adapt_every: int = 2,
    ):
        self.robust = bool(robust)
        self.safety = bool(safety)
        self.reference = float(reference)
        self.admm_iterations = int(admm_iterations)
        self.prediction_steps = int(prediction_steps)
        self.adapt_every = max(1, int(adapt_every))
        self.name = "RSP-ADMM-PID (proposed)" if self.robust and self.safety else "ADMM-PID"

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject
        self.dt = float(control_dt)
        self.model = replace(
            nominal_subject(subject.cohort),
            basal_rate=subject.basal_rate,
            max_rate=subject.max_rate,
        )
        self.observer = MinimalEKF(self.model, self.dt)
        self.gain_nominal = NOMINAL_GAINS[subject.cohort].copy()
        self.theta_nominal = (self.gain_nominal - GAIN_MIN) / (GAIN_MAX - GAIN_MIN)
        self.theta_prev = self.theta_nominal.copy()
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.first = True
        self.step_count = 0

    @staticmethod
    def _project(theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, 0.0, 1.0)

    @staticmethod
    def _gains(theta: np.ndarray) -> np.ndarray:
        return GAIN_MIN + np.clip(theta, 0.0, 1.0) * (GAIN_MAX - GAIN_MIN)

    def _scenario_set(self) -> list[tuple[float, float]]:
        if not self.robust:
            return [(0.0, 1.0)]
        # Unknown meal appearance (mg/dL/min) and insulin-sensitivity scale.
        return [(0.0, 1.25), (2.0, 1.0), (4.2, 0.82)]

    def _rollout_cost(
        self,
        theta: np.ndarray,
        x0: np.ndarray,
        integral0: float,
        derivative0: float,
        error0: float,
        t_min: float,
    ) -> float:
        gains = self._gains(theta)
        scenario_costs = []
        for appearance0, sensitivity in self._scenario_set():
            x = x0.copy()
            integ = float(integral0)
            deriv = float(derivative0)
            prev_error = float(error0)
            value = 0.0
            for j in range(self.prediction_steps):
                error = float(x[0] - self.reference)
                integ = float(np.clip(integ + self.dt * error, -12_000.0, 12_000.0))
                d_raw = (error - prev_error) / self.dt
                deriv = 0.67 * deriv + 0.33 * d_raw
                rate = self.model.basal_rate + float(gains @ np.array([error, integ, deriv]))
                rate = float(np.clip(rate, 0.0, self.model.max_rate))
                appearance = appearance0 * np.exp(-j * self.dt / 45.0)
                x = advance_interval(
                    t_min + j * self.dt,
                    x,
                    rate,
                    self.model,
                    None,
                    self.dt,
                    self.dt,
                    appearance_override=float(appearance),
                    sensitivity_scale=float(sensitivity),
                )
                g = float(x[0])
                track = ((g - self.reference) / 34.0) ** 2
                tight_high = 0.90 * (max(g - 140.0, 0.0) / 40.0) ** 2
                hypo = 9.0 * (max(75.0 - g, 0.0) / 18.0) ** 2
                hyper = 2.0 * (max(g - 180.0, 0.0) / 60.0) ** 2
                effort = 0.010 * ((rate - self.model.basal_rate) / self.model.max_rate) ** 2
                value += track + tight_high + hypo + hyper + effort
                prev_error = error
            scenario_costs.append(value / self.prediction_steps)
        costs = np.asarray(scenario_costs)
        aggregate = float(np.mean(costs))
        if self.robust:
            aggregate += 0.20 * float(np.max(costs))
        return aggregate

    def _finite_difference_gradient(
        self,
        theta: np.ndarray,
        xhat: np.ndarray,
        integral: float,
        derivative: float,
        error: float,
        t_min: float,
    ) -> np.ndarray:
        gradient = np.zeros(3)
        delta = 0.018
        for idx in range(3):
            plus = theta.copy()
            minus = theta.copy()
            plus[idx] = min(1.0, plus[idx] + delta)
            minus[idx] = max(0.0, minus[idx] - delta)
            denominator = plus[idx] - minus[idx]
            if denominator <= 1e-12:
                continue
            jp = self._rollout_cost(plus, xhat, integral, derivative, error, t_min)
            jm = self._rollout_cost(minus, xhat, integral, derivative, error, t_min)
            gradient[idx] = (jp - jm) / denominator
        return np.clip(gradient, -20.0, 20.0)

    def _admm_update(
        self,
        xhat: np.ndarray,
        integral: float,
        derivative: float,
        error: float,
        t_min: float,
    ) -> np.ndarray:
        z = self.theta_prev.copy()
        y = self.theta_prev.copy()
        dual = np.zeros(3)
        rho = 7.0
        smooth = 2.5
        nominal = 0.5
        step = 0.060 if self.robust else 0.055
        for _ in range(self.admm_iterations):
            grad = self._finite_difference_gradient(z, xhat, integral, derivative, error, t_min)
            z = self._project(z - step * (grad + rho * (z - y + dual)))
            y = self._project(
                (rho * (z + dual) + smooth * self.theta_prev + nominal * self.theta_nominal)
                / (rho + smooth + nominal)
            )
            dual = dual + z - y
        theta = self._project(0.5 * (z + y))
        # A small trust region prevents noisy CGM samples from causing abrupt gain jumps.
        radius = 0.12 if self.robust else 0.14
        theta = np.clip(theta, self.theta_prev - radius, self.theta_prev + radius)
        return self._project(theta)

    def _minimum_prediction(self, xhat: np.ndarray, rate: float, t_min: float) -> float:
        conservative = replace(self.model, p3=1.18 * self.model.p3, insulin_gain=1.10 * self.model.insulin_gain)
        x = xhat.copy()
        minimum = float(x[0])
        for j in range(9):
            x = advance_interval(
                t_min + j * self.dt,
                x,
                rate,
                conservative,
                None,
                self.dt,
                self.dt,
                appearance_override=0.0,
                sensitivity_scale=1.18,
            )
            minimum = min(minimum, float(x[0]))
        return minimum

    def _safety_projection(
        self, xhat: np.ndarray, nominal_rate: float, t_min: float, derivative: float
    ) -> float:
        if not self.safety:
            return nominal_rate
        if xhat[0] <= 78.0:
            return 0.0
        # Do not suppress corrective insulin while glucose is clearly high or
        # rising with ordinary IOB.  The predictive projection is activated in
        # the clinically relevant descending/near-range regime.
        basal_iob = self.model.basal_rate * self.model.tau_iob / 60.0
        if xhat[0] >= 125.0 or (xhat[0] >= 100.0 and derivative >= 0.0 and xhat[3] < basal_iob + 1.5):
            return nominal_rate
        safety_floor = 75.0
        if self._minimum_prediction(xhat, nominal_rate, t_min) >= safety_floor:
            return nominal_rate
        if self._minimum_prediction(xhat, 0.0, t_min) < safety_floor:
            return 0.0
        low, high = 0.0, float(nominal_rate)
        for _ in range(12):
            mid = 0.5 * (low + high)
            if self._minimum_prediction(xhat, mid, t_min) >= safety_floor:
                low = mid
            else:
                high = mid
        return low

    def act(self, t_min: float, glucose_cgm: float):
        xhat = self.observer.update(glucose_cgm)
        error = float(glucose_cgm - self.reference)
        derivative_raw = 0.0 if self.first else (error - self.prev_error) / self.dt
        self.derivative = 0.67 * self.derivative + 0.33 * derivative_raw
        candidate_integral = float(np.clip(self.integral + self.dt * error, -12_000.0, 12_000.0))
        if self.step_count % self.adapt_every == 0:
            theta = self._admm_update(
                xhat, candidate_integral, self.derivative, error, float(t_min)
            )
        else:
            theta = self.theta_prev.copy()
        gains = self._gains(theta)
        raw = self.subject.basal_rate + float(
            gains @ np.array([error, candidate_integral, self.derivative])
        )
        nominal_rate = float(np.clip(raw, 0.0, self.subject.max_rate))
        rate = float(
            self._safety_projection(xhat, nominal_rate, float(t_min), self.derivative)
        )
        intervened = rate < nominal_rate - 1e-4

        saturated_high = raw > self.subject.max_rate and error > 0.0
        saturated_low = raw < 0.0 and error < 0.0
        if not (saturated_high or saturated_low or intervened):
            self.integral = candidate_integral
        self.prev_error = error
        self.theta_prev = theta
        self.first = False
        self.step_count += 1
        self.observer.set_command(rate)
        return rate, {"gains": gains.copy(), "safety_intervened": intervened}


def policy_observation(
    *,
    glucose: float,
    derivative: float,
    integral: float,
    iob: float,
    previous_rate: float,
    t_min: float,
    subject: Subject,
) -> np.ndarray:
    """Shared normalized observation for direct PPO and PPO-PID."""

    one_hot = np.array(
        [float(subject.cohort == "t1d"), float(subject.cohort == "t2d")], dtype=float
    )
    return np.concatenate(
        [
            np.array(
                [
                    np.clip((glucose - 110.0) / 100.0, -1.5, 3.0),
                    np.clip(derivative / 8.0, -2.0, 2.0),
                    np.clip(integral / 8_000.0, -1.5, 1.5),
                    np.clip(iob / 6.0, 0.0, 3.0),
                    np.clip(previous_rate / max(subject.max_rate, 1.0), 0.0, 1.0),
                    np.sin(2.0 * np.pi * t_min / 1440.0),
                    np.cos(2.0 * np.pi * t_min / 1440.0),
                    subject.basal_rate / 2.0,
                ]
            ),
            one_hot,
        ]
    )


class DirectPolicyController:
    """Wrapper around a trained continuous policy that outputs pump rate."""

    name = "Direct PPO"

    def __init__(self, policy, reference: float = 110.0):
        self.policy = policy
        self.reference = float(reference)

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject
        self.dt = float(control_dt)
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.rate = subject.basal_rate
        self.iob = subject.basal_rate * subject.tau_iob / 60.0
        self.first = True

    def act(self, t_min: float, glucose_cgm: float):
        error = float(glucose_cgm - self.reference)
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / self.dt
        self.derivative = 0.67 * self.derivative + 0.33 * raw_derivative
        self.integral = float(np.clip(self.integral + error * self.dt, -12_000.0, 12_000.0))
        decay = np.exp(-self.dt / self.subject.tau_iob)
        self.iob = decay * self.iob + self.rate * self.subject.tau_iob / 60.0 * (1.0 - decay)
        obs = policy_observation(
            glucose=glucose_cgm,
            derivative=self.derivative,
            integral=self.integral,
            iob=self.iob,
            previous_rate=self.rate,
            t_min=t_min,
            subject=self.subject,
        )
        normalized = float(np.asarray(self.policy.deterministic_action(obs)).reshape(-1)[0])
        self.rate = float(np.clip(normalized, 0.0, 1.0) * self.subject.max_rate)
        self.prev_error = error
        self.first = False
        return self.rate, {"safety_intervened": False}


class PolicyPIDController:
    """PPO supervisor that maps observations to time-varying PID gains."""

    name = "PPO-PID"

    def __init__(self, policy, reference: float = 110.0):
        self.policy = policy
        self.reference = float(reference)

    def reset(self, subject: Subject, control_dt: float) -> None:
        self.subject = subject
        self.dt = float(control_dt)
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.rate = subject.basal_rate
        self.iob = subject.basal_rate * subject.tau_iob / 60.0
        self.first = True

    def act(self, t_min: float, glucose_cgm: float):
        error = float(glucose_cgm - self.reference)
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / self.dt
        self.derivative = 0.67 * self.derivative + 0.33 * raw_derivative
        candidate_integral = float(np.clip(self.integral + error * self.dt, -12_000.0, 12_000.0))
        decay = np.exp(-self.dt / self.subject.tau_iob)
        self.iob = decay * self.iob + self.rate * self.subject.tau_iob / 60.0 * (1.0 - decay)
        obs = policy_observation(
            glucose=glucose_cgm,
            derivative=self.derivative,
            integral=candidate_integral,
            iob=self.iob,
            previous_rate=self.rate,
            t_min=t_min,
            subject=self.subject,
        )
        normalized = np.asarray(self.policy.deterministic_action(obs), dtype=float).reshape(3)
        gains = GAIN_MIN + np.clip(normalized, 0.0, 1.0) * (GAIN_MAX - GAIN_MIN)
        raw = self.subject.basal_rate + float(
            gains @ np.array([error, candidate_integral, self.derivative])
        )
        self.rate = float(np.clip(raw, 0.0, self.subject.max_rate))
        if not ((raw > self.subject.max_rate and error > 0) or (raw < 0 and error < 0)):
            self.integral = candidate_integral
        self.prev_error = error
        self.first = False
        return self.rate, {"gains": gains.copy(), "safety_intervened": False}
