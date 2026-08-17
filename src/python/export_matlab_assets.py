#!/usr/bin/env python3
"""Export selected NumPy PPO actors to MATLAB-readable structures."""

from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "policies"
DESTINATION = ROOT / "matlab" / "policies"


def export(source_name: str, destination_name: str) -> None:
    data = np.load(SOURCE / source_name)
    policy = {
        "observation_dim": int(data["observation_dim"]),
        "action_dim": int(data["action_dim"]),
        "hidden_dim": int(data["hidden_dim"]),
        "w1": np.asarray(data["w1"], dtype=float),
        "b1": np.asarray(data["b1"], dtype=float),
        "w2": np.asarray(data["w2"], dtype=float),
        "b2": np.asarray(data["b2"], dtype=float),
        "log_std": np.asarray(data["log_std"], dtype=float),
    }
    savemat(DESTINATION / destination_name, {"policy": policy}, do_compression=True)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    export("direct_ppo_selected.npz", "direct_ppo_selected.mat")
    export("pid_ppo_selected.npz", "pid_ppo_selected.mat")
    print(f"Exported selected policies to {DESTINATION}")


if __name__ == "__main__":
    main()
