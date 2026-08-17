"""A compact, auditable NumPy implementation of clipped PPO.

The actor is a Gaussian MLP in an unconstrained latent space.  A logistic map
converts latent actions to ``[0, 1]`` for pump rate or PID-gain scaling.  The
implementation intentionally avoids external deep-learning frameworks so the
GitHub package can reproduce training in the restricted conference artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float):
        self.lr = float(learning_rate)
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.t = 0

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        total_norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        scale = min(1.0, 0.8 / max(total_norm, 1e-12))
        for key in parameters:
            grad = gradients[key] * scale
            self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * grad
            self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * grad * grad
            m_hat = self.m[key] / (1.0 - beta1**self.t)
            v_hat = self.v[key] / (1.0 - beta2**self.t)
            parameters[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


class GaussianMLPPolicy:
    """One-hidden-layer Gaussian policy and deterministic logistic action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        seed: int,
        hidden_dim: int = 32,
        output_bias: np.ndarray | None = None,
    ):
        rng = np.random.default_rng(int(seed))
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.parameters = {
            "w1": rng.normal(0.0, np.sqrt(1.0 / observation_dim), (observation_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": rng.normal(0.0, 0.025, (hidden_dim, action_dim)),
            "b2": np.zeros(action_dim) if output_bias is None else np.asarray(output_bias, dtype=float).copy(),
            "log_std": np.full(action_dim, -0.65),
        }

    def forward(self, observations: np.ndarray):
        obs = np.atleast_2d(np.asarray(observations, dtype=float))
        hidden = np.tanh(obs @ self.parameters["w1"] + self.parameters["b1"])
        mean = hidden @ self.parameters["w2"] + self.parameters["b2"]
        return mean, hidden, obs

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        mean, _, _ = self.forward(np.asarray(observation, dtype=float))
        return sigmoid(mean[0])

    def sample(self, observations: np.ndarray, rng: np.random.Generator):
        mean, _, _ = self.forward(observations)
        std = np.exp(self.parameters["log_std"])
        latent = mean + rng.normal(size=mean.shape) * std
        log_probability = self.log_probability(latent, mean)
        return sigmoid(latent), latent, log_probability

    def log_probability(self, latent: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
        latent = np.atleast_2d(np.asarray(latent, dtype=float))
        if mean is None:
            raise ValueError("mean must be supplied when evaluating latent actions")
        variance = np.exp(2.0 * self.parameters["log_std"])
        return -0.5 * np.sum(
            (latent - mean) ** 2 / variance
            + 2.0 * self.parameters["log_std"]
            + np.log(2.0 * np.pi),
            axis=1,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            **self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianMLPPolicy":
        data = np.load(path)
        obj = cls(
            int(data["observation_dim"]),
            int(data["action_dim"]),
            seed=0,
            hidden_dim=int(data["hidden_dim"]),
        )
        for key in obj.parameters:
            obj.parameters[key][...] = data[key]
        return obj


class ValueMLP:
    def __init__(self, observation_dim: int, seed: int, hidden_dim: int = 32):
        rng = np.random.default_rng(int(seed))
        self.parameters = {
            "w1": rng.normal(0.0, np.sqrt(1.0 / observation_dim), (observation_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": rng.normal(0.0, 0.05, (hidden_dim, 1)),
            "b2": np.zeros(1),
        }

    def forward(self, observations: np.ndarray):
        obs = np.atleast_2d(np.asarray(observations, dtype=float))
        hidden = np.tanh(obs @ self.parameters["w1"] + self.parameters["b1"])
        value = (hidden @ self.parameters["w2"] + self.parameters["b2"]).reshape(-1)
        return value, hidden, obs


@dataclass
class PPOBatch:
    observations: np.ndarray
    latent_actions: np.ndarray
    old_log_probability: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PPOAgent:
    """Clipped PPO with generalized advantage estimation and Adam updates."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        seed: int,
        *,
        output_bias: np.ndarray | None = None,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 8.0e-4,
        clip_ratio: float = 0.20,
        entropy_coefficient: float = 0.002,
    ):
        self.policy = GaussianMLPPolicy(
            observation_dim, action_dim, seed, output_bias=output_bias
        )
        self.value = ValueMLP(observation_dim, seed + 79)
        self.actor_optimizer = Adam(self.policy.parameters, actor_lr)
        self.critic_optimizer = Adam(self.value.parameters, critic_lr)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coefficient = float(entropy_coefficient)
        self.rng = np.random.default_rng(int(seed) + 101)

    def values(self, observations: np.ndarray) -> np.ndarray:
        return self.value.forward(observations)[0]

    def action(self, observation: np.ndarray):
        normalized, latent, log_probability = self.policy.sample(
            np.asarray(observation, dtype=float)[None, :], self.rng
        )
        value = self.values(np.asarray(observation, dtype=float)[None, :])[0]
        return normalized[0], latent[0], float(log_probability[0]), float(value)

    def update(self, batch: PPOBatch, epochs: int = 6, minibatch_size: int = 256) -> dict[str, float]:
        n = len(batch.observations)
        actor_losses, critic_losses, kls, clip_fractions = [], [], [], []
        for _ in range(int(epochs)):
            order = self.rng.permutation(n)
            for start in range(0, n, int(minibatch_size)):
                indices = order[start : start + int(minibatch_size)]
                obs = batch.observations[indices]
                latent = batch.latent_actions[indices]
                old_logp = batch.old_log_probability[indices]
                advantages = batch.advantages[indices]
                returns = batch.returns[indices]

                mean, hidden, obs_cache = self.policy.forward(obs)
                logp = self.policy.log_probability(latent, mean)
                ratio = np.exp(np.clip(logp - old_logp, -20.0, 20.0))
                positive = advantages >= 0.0
                active = (positive & (ratio <= 1.0 + self.clip_ratio)) | (
                    ~positive & (ratio >= 1.0 - self.clip_ratio)
                )
                clipped_ratio = np.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                objective = np.minimum(ratio * advantages, clipped_ratio * advantages)
                actor_loss = -float(np.mean(objective))
                dloss_dlogp = -active.astype(float) * ratio * advantages / len(indices)
                variance = np.exp(2.0 * self.policy.parameters["log_std"])
                difference = latent - mean
                grad_mean = dloss_dlogp[:, None] * difference / variance
                grad_log_std = np.sum(
                    dloss_dlogp[:, None] * (difference * difference / variance - 1.0), axis=0
                ) - self.entropy_coefficient
                grad_w2 = hidden.T @ grad_mean
                grad_b2 = np.sum(grad_mean, axis=0)
                grad_hidden = grad_mean @ self.policy.parameters["w2"].T
                grad_pre = grad_hidden * (1.0 - hidden * hidden)
                actor_gradients = {
                    "w1": obs_cache.T @ grad_pre,
                    "b1": np.sum(grad_pre, axis=0),
                    "w2": grad_w2,
                    "b2": grad_b2,
                    "log_std": grad_log_std,
                }
                self.actor_optimizer.step(self.policy.parameters, actor_gradients)

                predicted, v_hidden, v_obs = self.value.forward(obs)
                value_error = predicted - returns
                critic_loss = 0.5 * float(np.mean(value_error * value_error))
                grad_value = value_error[:, None] / len(indices)
                grad_v_w2 = v_hidden.T @ grad_value
                grad_v_b2 = np.sum(grad_value, axis=0)
                grad_v_hidden = grad_value @ self.value.parameters["w2"].T
                grad_v_pre = grad_v_hidden * (1.0 - v_hidden * v_hidden)
                critic_gradients = {
                    "w1": v_obs.T @ grad_v_pre,
                    "b1": np.sum(grad_v_pre, axis=0),
                    "w2": grad_v_w2,
                    "b2": grad_v_b2,
                }
                self.critic_optimizer.step(self.value.parameters, critic_gradients)

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                kls.append(float(np.mean(old_logp - logp)))
                clip_fractions.append(float(np.mean(np.abs(ratio - 1.0) > self.clip_ratio)))
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }


def generalized_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    next_values = np.asarray(next_values, dtype=float)
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * continuation - values[t]
        accumulator = delta + gamma * gae_lambda * continuation * accumulator
        advantages[t] = accumulator
    returns = advantages + values
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages, returns
