"""Train direct PPO and PPO-PID policies on randomized T1D/T2D subjects."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.python.controllers import GAIN_MAX, GAIN_MIN, NOMINAL_GAINS, policy_observation
from src.python.model import advance_interval, clinical_reward, initial_state, sample_day, sample_subject
from src.python.ppo_numpy import PPOAgent, PPOBatch, generalized_advantages, logit


OBSERVATION_DIM = 10


@dataclass
class EpisodeRecord:
    observations: list
    latent_actions: list
    log_probabilities: list
    values: list
    rewards: list
    dones: list
    next_values: list
    total_reward: float
    tir: float


class GlucoseTrainingEnvironment:
    def __init__(self, mode: str, control_dt: float = 5.0):
        if mode not in {"direct", "pid"}:
            raise ValueError("mode must be 'direct' or 'pid'")
        self.mode = mode
        self.control_dt = float(control_dt)
        self.horizon_steps = int(24 * 60 / self.control_dt)

    def reset(self, episode_seed: int, cohort: str) -> np.ndarray:
        self.subject = sample_subject(cohort, episode_seed)
        self.scenario = sample_day(episode_seed + 100_000, self.horizon_steps + 1)
        self.state = initial_state(self.subject, self.scenario)
        self.step_index = 0
        self.rate = self.subject.basal_rate
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.first = True
        self.glucose_history = [float(self.state[0])]
        return self._observation()

    def _measurement(self) -> float:
        k = self.step_index
        return float(
            np.clip(
                self.state[4]
                + self.subject.cgm_bias
                + self.subject.cgm_noise_sd * self.scenario.cgm_noise[k],
                40.0,
                400.0,
            )
        )

    def _observation(self) -> np.ndarray:
        glucose = self._measurement()
        error = glucose - 110.0
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / self.control_dt
        derivative = 0.67 * self.derivative + 0.33 * raw_derivative
        candidate_integral = float(
            np.clip(self.integral + error * self.control_dt, -12_000.0, 12_000.0)
        )
        return policy_observation(
            glucose=glucose,
            derivative=derivative,
            integral=candidate_integral,
            iob=float(self.state[3]),
            previous_rate=self.rate,
            t_min=self.step_index * self.control_dt,
            subject=self.subject,
        )

    def step(self, normalized_action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        glucose = self._measurement()
        error = glucose - 110.0
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / self.control_dt
        self.derivative = 0.67 * self.derivative + 0.33 * raw_derivative
        candidate_integral = float(
            np.clip(self.integral + error * self.control_dt, -12_000.0, 12_000.0)
        )
        action = np.clip(np.asarray(normalized_action, dtype=float), 0.0, 1.0)
        if self.mode == "direct":
            raw_rate = float(action[0] * self.subject.max_rate)
        else:
            gains = GAIN_MIN + action.reshape(3) * (GAIN_MAX - GAIN_MIN)
            raw_rate = self.subject.basal_rate + float(
                gains @ np.array([error, candidate_integral, self.derivative])
            )
        self.rate = float(np.clip(raw_rate, 0.0, self.subject.max_rate))
        if not (
            (raw_rate > self.subject.max_rate and error > 0.0)
            or (raw_rate < 0.0 and error < 0.0)
        ):
            self.integral = candidate_integral
        self.prev_error = error
        self.first = False

        t_min = self.step_index * self.control_dt
        self.state = advance_interval(
            t_min,
            self.state,
            self.rate,
            self.subject,
            self.scenario,
            self.control_dt,
            1.0,
        )
        self.step_index += 1
        self.state[0] = np.clip(
            self.state[0] + self.scenario.process_noise[self.step_index], 35.0, 500.0
        )
        self.glucose_history.append(float(self.state[0]))
        reward = clinical_reward(float(self.state[0]), self.rate, self.subject)
        done = self.step_index >= self.horizon_steps
        if done:
            observation = np.zeros(OBSERVATION_DIM)
        else:
            observation = self._observation()
        return observation, float(reward), bool(done)


def collect_episode(
    agent: PPOAgent, environment: GlucoseTrainingEnvironment, episode_seed: int, cohort: str
) -> EpisodeRecord:
    observation = environment.reset(episode_seed, cohort)
    observations, latents, logps, values, rewards, dones, next_values = [], [], [], [], [], [], []
    done = False
    while not done:
        action, latent, logp, value = agent.action(observation)
        next_observation, reward, done = environment.step(action)
        next_value = 0.0 if done else float(agent.values(next_observation[None, :])[0])
        observations.append(observation)
        latents.append(latent)
        logps.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        next_values.append(next_value)
        observation = next_observation
    glucose = np.asarray(environment.glucose_history)
    return EpisodeRecord(
        observations=observations,
        latent_actions=latents,
        log_probabilities=logps,
        values=values,
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        total_reward=float(np.sum(rewards)),
        tir=100.0 * float(np.mean((glucose >= 70.0) & (glucose <= 180.0))),
    )


def evaluate_deterministic(policy, environment: GlucoseTrainingEnvironment) -> tuple[float, float]:
    """Evaluate the current policy on four fixed, unseen validation episodes."""

    returns, tirs = [], []
    validation_cases = [
        (730_001, "t1d"),
        (730_099, "t1d"),
        (730_197, "t2d"),
        (730_293, "t2d"),
    ]
    for episode_seed, cohort in validation_cases:
        observation = environment.reset(episode_seed, cohort)
        done = False
        total = 0.0
        while not done:
            action = policy.deterministic_action(observation)
            observation, reward, done = environment.step(action)
            total += reward
        glucose = np.asarray(environment.glucose_history)
        returns.append(total)
        tirs.append(100.0 * float(np.mean((glucose >= 70.0) & (glucose <= 180.0))))
    return float(np.mean(returns)), float(np.mean(tirs))


def train_one(
    mode: str,
    seed: int,
    updates: int,
    episodes_per_update: int,
    output_dir: Path,
) -> dict:
    action_dim = 1 if mode == "direct" else 3
    if mode == "direct":
        output_bias = np.array([-1.15])
    else:
        nominal = 0.5 * (NOMINAL_GAINS["t1d"] + NOMINAL_GAINS["t2d"])
        output_bias = logit((nominal - GAIN_MIN) / (GAIN_MAX - GAIN_MIN))
    agent = PPOAgent(
        OBSERVATION_DIM,
        action_dim,
        seed,
        output_bias=output_bias,
        actor_lr=2.5e-4 if mode == "direct" else 2.0e-4,
        critic_lr=7.0e-4,
    )
    env = GlucoseTrainingEnvironment(mode)
    history = {
        "mode": mode,
        "seed": int(seed),
        "episode_return_mean": [],
        "episode_return_std": [],
        "tir_mean": [],
        "tir_std": [],
        "actor_loss": [],
        "critic_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
        "validation_return": [],
        "validation_tir": [],
    }
    best_validation_return = -np.inf
    best_parameters = {key: value.copy() for key, value in agent.policy.parameters.items()}
    best_update = 0
    for update in range(int(updates)):
        records = []
        for episode in range(int(episodes_per_update)):
            cohort = "t1d" if (update * episodes_per_update + episode) % 2 == 0 else "t2d"
            episode_seed = int(seed * 1_000_000 + update * 10_000 + episode * 97 + 17)
            records.append(collect_episode(agent, env, episode_seed, cohort))

        observations = np.asarray([x for r in records for x in r.observations])
        latents = np.asarray([x for r in records for x in r.latent_actions])
        logps = np.asarray([x for r in records for x in r.log_probabilities])
        values = np.asarray([x for r in records for x in r.values])
        rewards = np.asarray([x for r in records for x in r.rewards])
        dones = np.asarray([x for r in records for x in r.dones])
        next_values = np.asarray([x for r in records for x in r.next_values])
        advantages, returns = generalized_advantages(rewards, values, dones, next_values)
        diagnostics = agent.update(
            PPOBatch(observations, latents, logps, advantages, returns),
            epochs=5,
            minibatch_size=256,
        )
        returns_ep = np.array([r.total_reward for r in records])
        tir_ep = np.array([r.tir for r in records])
        history["episode_return_mean"].append(float(np.mean(returns_ep)))
        history["episode_return_std"].append(float(np.std(returns_ep, ddof=1)))
        history["tir_mean"].append(float(np.mean(tir_ep)))
        history["tir_std"].append(float(np.std(tir_ep, ddof=1)))
        for key, value in diagnostics.items():
            history[key].append(float(value))
        validation_return, validation_tir = evaluate_deterministic(agent.policy, env)
        history["validation_return"].append(validation_return)
        history["validation_tir"].append(validation_tir)
        if validation_return > best_validation_return:
            best_validation_return = validation_return
            best_parameters = {key: value.copy() for key, value in agent.policy.parameters.items()}
            best_update = update + 1
        print(
            f"{mode} seed={seed} update={update + 1:02d}/{updates} "
            f"return={np.mean(returns_ep):8.1f} TIR={np.mean(tir_ep):5.1f}% "
            f"valTIR={validation_tir:5.1f}% KL={diagnostics['approx_kl']:.4f}",
            flush=True,
        )

    for key, value in best_parameters.items():
        agent.policy.parameters[key][...] = value
    history["best_update"] = int(best_update)
    history["best_validation_return"] = float(best_validation_return)
    stem = f"{mode}_ppo_seed{seed}"
    agent.policy.save(output_dir / f"{stem}.npz")
    with (output_dir / f"{stem}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "pid", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--episodes-per-update", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "policies")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["direct", "pid"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for seed in args.seeds:
            train_one(mode, seed, args.updates, args.episodes_per_update, args.output_dir)


if __name__ == "__main__":
    main()
