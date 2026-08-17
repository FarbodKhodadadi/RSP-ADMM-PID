"""Generate paper and supplementary figures (vector PDF + 600-dpi PNG)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PROPOSED = "RSP-ADMM-PID (proposed)"
METHODS = ["Fixed PID", "SL-LQG", "Direct PPO", "PPO-PID", "ADMM-PID", PROPOSED]
KEYS = {
    "Fixed PID": "fixed_pid",
    "SL-LQG": "sl_lqg",
    "Direct PPO": "direct_ppo",
    "PPO-PID": "ppo_pid",
    "ADMM-PID": "admm_pid",
    PROPOSED: "rsp_admm_pid",
}
COLORS = {
    "Normal physiology": "#111111",
    "Fixed PID": "#0072B2",
    "SL-LQG": "#E69F00",
    "Direct PPO": "#7A7A7A",
    "PPO-PID": "#009E73",
    "ADMM-PID": "#CC79A7",
    PROPOSED: "#D55E00",
}
LINESTYLES = {
    "Normal physiology": "--",
    "Fixed PID": "-",
    "SL-LQG": ":",
    "Direct PPO": (0, (5, 2)),
    "PPO-PID": "-.",
    "ADMM-PID": (0, (3, 1, 1, 1)),
    PROPOSED: "-",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8.2, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
        return patch

    def arrow(x1, y1, x2, y2, label="", rad=0.0):
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#333333",
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", va="bottom", fontsize=7.2)

    box(0.35, 1.45, 2.05, 1.15, "Virtual cohort\nNormal / T1D / T2D", "#E8F0F7", weight="bold")
    box(0.35, 3.08, 2.05, 0.58, "Meals, exercise, uncertainty", "#F3EEE5", fontsize=7.0)
    box(0.35, 0.28, 2.05, 0.58, "CGM lag, bias, noise", "#F3EEE5", fontsize=7.0)
    box(3.15, 1.55, 1.40, 0.95, "State\nobserver", "#EAF4EA", fontsize=7.6)
    box(5.05, 1.55, 1.85, 0.95, "Scenario ADMM\ngain update", "#FCE9E2", fontsize=7.4, weight="bold")
    box(7.40, 1.55, 1.25, 0.95, "Adaptive\nPID", "#FCE9E2", fontsize=7.6, weight="bold")
    box(9.15, 1.55, 1.90, 0.95, "Predictive safety\nprojection", "#FCE9E2", fontsize=7.2, weight="bold")
    box(11.65, 1.55, 1.45, 0.95, "Insulin\npump", "#E8F0F7", fontsize=7.6, weight="bold")

    arrow(2.40, 2.02, 3.15, 2.02)
    arrow(4.55, 2.02, 5.05, 2.02)
    arrow(6.90, 2.02, 7.40, 2.02)
    arrow(8.65, 2.02, 9.15, 2.02)
    arrow(11.05, 2.02, 11.65, 2.02)
    ax.text(2.77, 2.23, "$G_{\\mathrm{CGM}}$", ha="center", fontsize=6.8)
    ax.text(4.80, 2.23, "$\\hat{x}_t$", ha="center", fontsize=6.8)
    ax.text(7.15, 2.23, "$K_t$", ha="center", fontsize=6.8)
    ax.text(8.90, 2.23, "$u_t^{\\mathrm{nom}}$", ha="center", fontsize=6.8)
    ax.text(11.35, 2.23, "$u_t^*$", ha="center", fontsize=6.8)
    arrow(1.38, 3.08, 1.38, 2.60)
    arrow(1.38, 1.45, 1.38, 0.86)
    # Curved pump-to-patient physiological loop, kept below the controller path.
    loop = FancyArrowPatch(
        (12.35, 1.52),
        (1.95, 1.42),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color="#333333",
        connectionstyle="arc3,rad=-0.24",
    )
    ax.add_patch(loop)
    ax.text(7.05, 0.23, "closed-loop insulin action", ha="center", va="center", fontsize=6.9)
    ax.text(8.05, 0.91, "RSP-ADMM-PID (proposed)", ha="center", va="center", fontsize=8.1, weight="bold", color="#A64015")
    ax.plot([4.85, 11.20], [1.16, 1.16], color="#D55E00", lw=1.0)
    save(fig, "fig1_architecture")


def training_curves() -> None:
    histories = {}
    for mode in ("direct", "pid"):
        histories[mode] = [
            json.load((RESULTS / "policies" / f"{mode}_ppo_seed{seed}_history.json").open())
            for seed in (11, 29, 47)
        ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), constrained_layout=True)
    for mode, label, color in (("direct", "Direct PPO", COLORS["Direct PPO"]), ("pid", "PPO-PID", COLORS["PPO-PID"])):
        values = np.asarray([h["validation_return"] for h in histories[mode]], dtype=float)
        tir = np.asarray([h["validation_tir"] for h in histories[mode]], dtype=float)
        updates = np.arange(1, values.shape[1] + 1)
        for ax, data, ylabel in (
            (axes[0], values, "Validation return"),
            (axes[1], tir, "Validation TIR (%)"),
        ):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1)
            ax.fill_between(updates, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
            ax.plot(updates, mean, color=color, lw=1.55, label=label)
            ax.set_xlabel("PPO update")
            ax.set_ylabel(ylabel)
            ax.grid(True)
    axes[0].set_title("(a) Fixed validation return, mean $\\pm$ SD")
    axes[1].set_title("(b) Fixed validation time in 70--180 mg/dL")
    axes[1].axhline(70.0, color="#555555", lw=0.8, ls=":", label="Consensus target")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, "fig2_ppo_training")


def _stack(data, cohort: str, method: str, quantity: str, count: int = 30) -> np.ndarray:
    key = KEYS[method]
    return np.stack([data[f"{cohort}_{key}_{index:02d}_{quantity}"] for index in range(count)])


def glucose_trajectories(data) -> None:
    time_h = data["time_min"] / 60.0
    normal = np.stack([data[f"normal_normal_glucose_{i:02d}"] for i in range(30)])
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.35), sharex=True)
    for ax, cohort, title in zip(axes, ("t1d", "t2d"), ("Type-1 virtual cohort", "Type-2 virtual cohort")):
        ax.axhspan(70, 180, color="#DDEEDB", alpha=0.45, zorder=0)
        for meal_time in (8.0, 13.0, 19.0):
            ax.axvline(meal_time, color="#6A6A6A", lw=0.55, ls=(0, (2, 2)), alpha=0.55, zorder=0)
        n_mean, n_std = normal.mean(axis=0), normal.std(axis=0, ddof=1)
        ax.fill_between(time_h, n_mean - n_std, n_mean + n_std, color=COLORS["Normal physiology"], alpha=0.06, linewidth=0)
        ax.plot(time_h, n_mean, color=COLORS["Normal physiology"], ls="--", lw=1.35, label="Normal physiology")
        for method in METHODS:
            values = _stack(data, cohort, method, "glucose")
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.fill_between(time_h, mean - std, mean + std, color=COLORS[method], alpha=0.055, linewidth=0)
            ax.plot(
                time_h,
                mean,
                color=COLORS[method],
                ls=LINESTYLES[method],
                lw=1.75 if method == PROPOSED else 1.10,
                label=method,
            )
        ax.axhline(70, color="#B2182B", ls=":", lw=0.8)
        ax.axhline(180, color="#B2182B", ls=":", lw=0.8)
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(title + ", $n=30$ (mean $\\pm$ SD)", pad=4)
        ax.set_ylim(55, 290)
        ax.grid(True)
    axes[-1].set_xlabel("Time (h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.86, bottom=0.09, hspace=0.24, left=0.085, right=0.99)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    save(fig, "fig3_glucose_mean_std")


def outcome_distributions(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    short = ["PID", "SL-LQG", "PPO", "PPO-PID", "ADMM", "RSP"]
    for col, cohort in enumerate(("t1d", "t2d")):
        for row, (metric, ylabel) in enumerate(
            (("tir_70_180", "TIR 70--180 (%)"), ("normality_rmse", "RMSE to matched normal (mg/dL)"))
        ):
            ax = axes[row, col]
            arrays = [
                raw[(raw.cohort == cohort) & (raw.method == method)][metric].to_numpy(float)
                for method in METHODS
            ]
            parts = ax.violinplot(arrays, positions=np.arange(1, 7), widths=0.75, showmeans=False, showextrema=False)
            for body, method in zip(parts["bodies"], METHODS):
                body.set_facecolor(COLORS[method])
                body.set_edgecolor(COLORS[method])
                body.set_alpha(0.22)
            box = ax.boxplot(
                arrays,
                positions=np.arange(1, 7),
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.7},
                capprops={"linewidth": 0.7},
                boxprops={"linewidth": 0.7},
            )
            for patch, method in zip(box["boxes"], METHODS):
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.55)
            rng = np.random.default_rng(1701 + 100 * col + row)
            for position, values, method in zip(np.arange(1, 7), arrays, METHODS):
                jitter = rng.uniform(-0.075, 0.075, len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=5.5,
                    color=COLORS[method],
                    alpha=0.52,
                    edgecolors="white",
                    linewidths=0.20,
                    zorder=3,
                )
            ax.set_xticks(np.arange(1, 7), short, rotation=32, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y")
            ax.set_title(("Type-1" if cohort == "t1d" else "Type-2") + f" cohort: {'TIR' if row == 0 else 'normality error'}")
    save(fig, "fig4_outcome_distributions")


def gain_trajectories(data) -> None:
    time_h = data["time_min"] / 60.0
    methods = ["PPO-PID", "ADMM-PID", PROPOSED]
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.15), sharex=True, constrained_layout=True)
    labels = ["$K_p$", "$K_i$", "$K_d$"]
    scales = [1.0, 1e4, 1.0]
    ylabels = ["$K_p$", "$10^4 K_i$", "$K_d$"]
    for col, cohort in enumerate(("t1d", "t2d")):
        for row in range(3):
            ax = axes[row, col]
            for method in methods:
                values = _stack(data, cohort, method, "gains")[:, :, row] * scales[row]
                mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)
                ax.fill_between(time_h, mean - std, mean + std, color=COLORS[method], alpha=0.10, linewidth=0)
                ax.plot(
                    time_h,
                    mean,
                    color=COLORS[method],
                    ls=LINESTYLES[method],
                    lw=1.6 if method == PROPOSED else 1.1,
                    label=method,
                )
            ax.set_ylabel(ylabels[row])
            ax.grid(True)
            if row == 0:
                ax.set_title(("Type-1" if cohort == "t1d" else "Type-2") + " gain adaptation, mean $\\pm$ SD")
            if row == 2:
                ax.set_xlabel("Time (h)")
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    save(fig, "fig5_adaptive_gains")


def safety_example(data) -> None:
    index = 10
    time_h = data["time_min"] / 60.0
    normal = data[f"normal_normal_glucose_{index:02d}"]
    admm_g = data[f"t1d_admm_pid_{index:02d}_glucose"]
    rsp_g = data[f"t1d_rsp_admm_pid_{index:02d}_glucose"]
    admm_u = data[f"t1d_admm_pid_{index:02d}_rate"]
    rsp_u = data[f"t1d_rsp_admm_pid_{index:02d}_rate"]
    safety = data[f"t1d_rsp_admm_pid_{index:02d}_safety"].astype(bool)
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.75), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.axhspan(70, 180, color="#DDEEDB", alpha=0.45)
    ax.plot(time_h, normal, color="#111111", ls="--", lw=1.0, label="Matched normal")
    ax.plot(time_h, admm_g, color=COLORS["ADMM-PID"], lw=1.2, label="ADMM-PID")
    ax.plot(time_h, rsp_g, color=COLORS[PROPOSED], lw=1.7, label="RSP-ADMM-PID")
    ax.axhline(70, color="#B2182B", ls=":", lw=0.8)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("High-sensitivity held-out type-1 subject (seed index 10)")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.grid(True)
    ax = axes[1]
    ax.step(time_h, admm_u, where="post", color=COLORS["ADMM-PID"], lw=1.05, label="ADMM-PID")
    ax.step(time_h, rsp_u, where="post", color=COLORS[PROPOSED], lw=1.45, label="RSP-ADMM-PID")
    ax.scatter(time_h[safety], rsp_u[safety], marker="v", s=15, facecolor="none", edgecolor="#B2182B", linewidth=0.8, label="Safety projection active")
    ax.set_ylabel("Pump rate (U/h)")
    ax.set_xlabel("Time (h)")
    ax.grid(True)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    save(fig, "fig6_safety_example")


def cohort_distributions() -> None:
    # Generate a large deterministic sample only for visualizing the declared
    # domain-randomization distributions; these are not extra test subjects.
    from src.python.model import sample_subject

    frames = []
    for cohort in ("normal", "t1d", "t2d"):
        for i in range(1_000):
            s = sample_subject(cohort, 2_000_000 + 31 * i)
            frames.append(
                {
                    "cohort": cohort,
                    "p1": s.p1,
                    "p3": s.p3 * 1e5,
                    "secretion": s.secretion_gain,
                    "tau_iob": s.tau_iob,
                }
            )
    frame = pd.DataFrame(frames)
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.25), constrained_layout=True)
    specifications = [
        ("p1", "$p_1$ (min$^{-1}$)"),
        ("p3", "$10^5 p_3$"),
        ("secretion", "$\\kappa_{sec}$"),
        ("tau_iob", "$\\tau_{IOB}$ (min)"),
    ]
    colors = ["#4C78A8", "#D55E00", "#009E73"]
    for ax, (metric, label) in zip(axes, specifications):
        arrays = [frame[frame.cohort == cohort][metric].to_numpy() for cohort in ("normal", "t1d", "t2d")]
        parts = ax.violinplot(arrays, showmeans=False, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        ax.boxplot(arrays, widths=0.22, showfliers=False, medianprops={"color": "black"})
        ax.set_xticks([1, 2, 3], ["Normal", "T1D", "T2D"], rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.grid(True, axis="y")
    save(fig, "figS1_cohort_parameter_distributions")


def uncertainty_sensitivity(raw: pd.DataFrame) -> None:
    """Exploratory rank sensitivity on the held-out proposed-controller cases."""

    from src.python.model import nominal_subject, sample_day, sample_subject

    rows = []
    for cohort in ("t1d", "t2d"):
        nominal = nominal_subject(cohort)
        outcomes = raw[(raw.cohort == cohort) & (raw.method == PROPOSED)].set_index("seed_index")
        for index in range(30):
            subject = sample_subject(cohort, 900_000 + 17 * index)
            day = sample_day(1_000_000 + 31 * index)
            outcome = outcomes.loc[index]
            rows.append(
                {
                    "cohort": cohort,
                    "p1": subject.p1 / nominal.p1,
                    "p3": subject.p3 / nominal.p3,
                    "insulin_gain": subject.insulin_gain / nominal.insulin_gain,
                    "tau_iob": subject.tau_iob / nominal.tau_iob,
                    "cgm_noise": subject.cgm_noise_sd,
                    "cgm_bias": subject.cgm_bias,
                    "carbohydrate": float(np.sum(day.meal_carbs_g)),
                    "activity": float(day.exercise_duration * day.exercise_intensity),
                    "tir": outcome.tir_70_180,
                    "normality": outcome.normality_rmse,
                    "insulin": outcome.total_insulin_u,
                    "rmse": outcome.rmse_110,
                }
            )
    frame = pd.DataFrame(rows)
    features = ["p1", "p3", "insulin_gain", "tau_iob", "cgm_noise", "cgm_bias", "carbohydrate", "activity"]
    feature_labels = ["$p_1$", "$p_3$", "$k_u$", "$\\tau_{IOB}$", "$\\sigma_{CGM}$", "$b_{CGM}$", "Total CHO", "Activity dose"]
    targets = ["tir", "normality", "insulin", "rmse"]
    target_labels = ["TIR", "Normality RMSE", "Insulin", "RMSE$_{110}$"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25), constrained_layout=True)
    image_handle = None
    for ax, cohort, title in zip(axes, ("t1d", "t2d"), ("Type-1", "Type-2")):
        subset = frame[frame.cohort == cohort]
        matrix = np.zeros((len(features), len(targets)))
        for row, feature in enumerate(features):
            for col, target in enumerate(targets):
                matrix[row, col] = spearmanr(subset[feature], subset[target]).statistic
        image_handle = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if abs(matrix[row, col]) > 0.55 else "#222222"
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color, fontsize=7.0)
        ax.set_xticks(range(len(targets)), target_labels, rotation=26, ha="right")
        ax.set_yticks(range(len(features)), feature_labels)
        ax.set_title(f"{title} held-out rank sensitivity ($n=30$)")
    cbar = fig.colorbar(image_handle, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Spearman $\\rho$")
    save(fig, "fig7_uncertainty_sensitivity")


def main() -> None:
    configure()
    raw = pd.read_csv(RESULTS / "per_subject_metrics.csv")
    data = np.load(RESULTS / "trajectories.npz")
    architecture()
    training_curves()
    glucose_trajectories(data)
    outcome_distributions(raw)
    gain_trajectories(data)
    safety_example(data)
    cohort_distributions()
    uncertainty_sensitivity(raw)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
