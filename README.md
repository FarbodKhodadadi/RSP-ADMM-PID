# RSP-ADMM-PID

## Robust Safety-Projected Adaptive Glucose Control Across Heterogeneous Virtual Cohorts

Reproducible source code, trained policies, seeded virtual-cohort experiments, held-out results, and MATLAB/Python implementations for the in-silico study:

> **RSP-ADMM-PID: Robust Safety-Projected Adaptive Glucose Control Across Heterogeneous Virtual Cohorts**

RSP-ADMM-PID is an adaptive glucose-control architecture combining:

* an extended Bergman-type physiological model,
* stochastic meals and activity,
* physiological parameter heterogeneity,
* CGM lag, bias, and noise,
* Extended Kalman Filter (EKF) state estimation,
* scenario-based robust prediction,
* bounded online PID-gain adaptation,
* a three-iteration linearized/inexact ADMM update, and
* a predictive low-glucose safety projection.

The repository is intended to make the complete simulation benchmark reproducible from source code and recorded experimental configurations.

> **This is an in-silico methodological benchmark. It is not a clinical trial, medical device, or insulin-dosing system.**

---

## Repository Structure

```text
RSP-ADMM-PID/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── python/
│   ├── ...
│
├── matlab/
│   ├── ...
│
├── results/
│   ├── experiment_manifest.json
│   ├── summary_mean_std.csv
│   ├── paired_wilcoxon_holm.csv
│   ├── ...
│
└── legacy/
    ├── ...
```

### Directory overview

| Path               | Purpose                                                                                                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `python/`          | Reference implementation, PPO training, controller implementations, experiments, statistics, and reproducibility checks |
| `matlab/`          | Human-readable MATLAB implementation/mirror of the simulator and controllers                                            |
| `results/`         | Archived subject-level results, statistical analyses, trajectories, policies, and experiment configuration              |
| `legacy/`          | Original submitted MATLAB implementation and provenance material                                                        |
| `requirements.txt` | Python dependencies                                                                                                     |
| `LICENSE`          | Software license                                                                                                        |

The manuscript and publication figures are intentionally **not included in this repository version**.

---

# Method

The complete closed-loop architecture is:

```text
                    Virtual Patient
                          │
                          ▼
                ┌───────────────────┐
                │ Extended Bergman  │
                │ Physiological     │
                │ Model             │
                └─────────┬─────────┘
                          │
                          ▼
                     CGM signal
                          │
                          ▼
                ┌───────────────────┐
                │       EKF         │
                │ State Estimation  │
                └─────────┬─────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Scenario Predictions  │
              │                       │
              │  3 uncertainty cases  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Linearized / Inexact  │
              │ ADMM Gain Update      │
              │                       │
              │       L = 3           │
              └───────────┬───────────┘
                          │
                          ▼
                  Adaptive PID
                          │
                          ▼
                Nominal insulin
                    command
                          │
                          ▼
              ┌───────────────────────┐
              │ Predictive Safety     │
              │ Projection            │
              │                       │
              │ 45-min rollout        │
              └───────────┬───────────┘
                          │
                          ▼
                  Applied insulin
                          │
                          ▼
                    Virtual Plant
```

The implementation distinguishes the optimization layer from the predictive safety layer. The mathematical formulation specifies three fixed-budget iterations of a **linearized/inexact ADMM** procedure rather than claiming convergence of ADMM to an optimum.

---

# Virtual Cohort

The benchmark contains three physiological populations:

* Normal physiology
* Type 1 diabetes (T1D)
* Insulin-treated type 2 diabetes (T2D)

The virtual cohort includes:

* lognormal inter-subject parameter variation,
* stochastic breakfast/lunch/dinner events,
* optional snacks,
* exercise,
* circadian modulation,
* CGM lag,
* CGM bias,
* CGM measurement noise,
* process noise,
* insulin-on-board dynamics, and
* pump command constraints.

Normal physiology is used as a **matched counterfactual reference**.

The normal reference receives the same declared meal/activity realization but does not represent a controller treatment arm.

---

# Proposed Controller

## 1. EKF State Estimation

The controller observes CGM measurements rather than the complete physiological state.

The EKF estimates the control-relevant state.

The reconstructed state is then passed to the scenario prediction and adaptive-control layers.

---

## 2. Scenario Prediction

At each control step, the proposed controller evaluates three short-horizon scenarios representing different combinations of:

* meal appearance,
* insulin sensitivity, and
* physiological uncertainty.

The formulation specifies three scenario rollouts corresponding to low-risk/nominal/high-risk physiological cases.

---

## 3. Adaptive PID

The controller maintains normalized PID gains:

```text
[Kp, Ki, Kd]
```

The gains are updated online.

A component-wise trust region limits the amount by which the gains can change at each update.

The resulting PID command is constrained by the pump limits.

---

## 4. ADMM Gain Optimization

The gain-selection problem is solved using:

```text
L = 3 ADMM iterations
```

The implementation uses:

* a primal gain update,
* auxiliary-variable update,
* scaled dual update,
* first-order/linearized primal optimization,
* gain constraints, and
* bounded gain adaptation.

This should be described as **linearized/inexact ADMM**, not exact/converged ADMM. The corrected mathematical formulation explicitly requires this distinction.

---

## 5. Predictive Safety Projection

The nominal PID command is evaluated over a predictive horizon.

When the predicted trajectory violates the declared low-glucose floor, the command is reduced through a one-dimensional projection.

The implementation uses bisection for the admissible command.

The mathematical formulation describes the projection as model-predicted feasibility, **not a clinical safety guarantee**.

---

# Controllers

The benchmark contains the following controller families:

| Controller   | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| Fixed PID    | Conventional fixed-gain baseline                           |
| SL-LQG       | State-linearized LQG baseline                              |
| Direct PPO   | RL controller directly generating normalized pump command  |
| PPO-PID      | RL controller generating PID gains                         |
| ADMM-PID     | Non-robust adaptive baseline                               |
| RSP-ADMM-PID | Proposed robust adaptive controller with safety projection |

The paper's experimental protocol uses the same subject/day/noise realization for each controller, enabling paired comparisons.

---

# PPO Baselines

The archived PPO configuration uses:

```text
Seeds:
11, 29, 47

Actor:
32-unit tanh network

Action:
Gaussian latent action
followed by sigmoid normalization

PPO:
epsilon = 0.2

GAE:
gamma = 0.995
lambda = 0.95

Epochs per update:
6

Training:
24 updates
6 randomized days per update
```

Two PPO action parameterizations are evaluated:

### Direct PPO

Maps a ten-dimensional observation to a normalized pump command.

### PPO-PID

Maps the observation to three normalized PID gains.

The best checkpoint for each training seed is selected using validation data before the held-out test set is evaluated. The test seeds are not used for controller tuning or checkpoint selection.

---

# Reproducibility

The archived subject seeds are:

```text
subject_seed(i) = 900000 + 17*i
```

for:

```text
i = 0, ..., 29
```

The day seeds are:

```text
day_seed(i) = 1000000 + 31*i
```

The complete archived configuration should be available in:

```text
results/experiment_manifest.json
```

This manifest should be treated as the authoritative record of the benchmark configuration.

---

# Python Reference Implementation

Python **3.12** was used for the archived benchmark.

Create the environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Smoke Test

Run:

```bash
python python/smoke_test.py
```

The smoke test should verify at minimum:

* model execution,
* finite physiological states,
* bounded insulin commands,
* controller execution,
* deterministic seeded behavior, and
* basic metric calculation.

---

# Train PPO Controllers

The archived PPO training command is:

```bash
python python/train_rl.py --mode both --seeds 11 29 47 \
  --updates 24 --episodes-per-update 6
```

This trains both:

```text
Direct PPO
PPO-PID
```

using the declared training seeds.

---

# Run the Complete Benchmark

Run:

```bash
python python/run_experiments.py
```

The experiment runner should:

1. generate/load the declared virtual subjects;
2. generate matched disturbances;
3. evaluate every controller;
4. preserve subject/day pairing;
5. calculate subject-level metrics;
6. perform ablation experiments;
7. perform statistical testing;
8. save raw and summarized results; and
9. record the experimental configuration.

---

# Generate Archived Analysis

If the repository contains the corresponding analysis scripts, run them from the generated results.

The archived statistical analysis includes:

* mean ± sample SD,
* paired comparisons,
* one-sided Wilcoxon signed-rank tests,
* Holm correction,
* paired bootstrap confidence intervals.

The reported analysis uses 5,000 bootstrap resamples for paired mean differences.

---

# MATLAB Implementation

The MATLAB implementation is located under:

```text
matlab/
```

MATLAB R2021b or newer is recommended.

Run the smoke test:

```matlab
cd matlab
run_smoke
```

Run the Monte Carlo benchmark:

```matlab
[raw, summary] = run_monte_carlo(30);
```

The MATLAB implementation is intended as a human-readable mirror of the model and controller logic.

### Random-number generators

MATLAB and NumPy use different random-number generators.

Therefore:

* MATLAB reproduces the declared mathematical model and distributions;
* Python is the reference implementation for reproducing the archived CSV results seed-for-seed.

Identical seed numbers do not imply identical random trajectories across MATLAB and Python.

---

# Archived Results

The `results/` directory contains the outputs used for the reported benchmark.

Important files include:

```text
results/
├── experiment_manifest.json
├── summary_mean_std.csv
├── paired_wilcoxon_holm.csv
├── ...
```

The repository should preserve the relationship:

```text
Configuration
      ↓
Seeds
      ↓
Virtual subjects
      ↓
Controller simulation
      ↓
Subject-level outcomes
      ↓
Statistical analysis
      ↓
Archived results
```

---

# Main Reported Performance

The archived benchmark evaluated 30 independently seeded subjects per disease cohort using paired 24-hour meal/activity realizations.

For RSP-ADMM-PID:

| Cohort | TIR 70–180 mg/dL | TBR <70 mg/dL | RMSE to matched normal |
| ------ | ---------------: | ------------: | ---------------------: |
| T1D    |    87.16 ± 5.31% |  0.00 ± 0.00% |     23.76 ± 8.71 mg/dL |
| T2D    |    89.85 ± 4.79% |  0.00 ± 0.00% |     17.50 ± 3.61 mg/dL |

Values are mean ± sample standard deviation.

The full comparator results should be taken from:

```text
results/summary_mean_std.csv
```

and the paired statistical results from:

```text
results/paired_wilcoxon_holm.csv
```

---

# Tests and Reproducibility Assertions

The archived methodology states that implementation-level assertions check:

* deterministic behavior,
* finite states,
* command bounds,
* seed separation, and
* consistency between tables and generated results.

These checks are important because this repository is intended as a reproducible computational benchmark rather than merely a controller demonstration.

---

# Scientific Limitations

This repository implements a compact extended-Bergman virtual cohort.

It is **not**:

* the proprietary UVA/Padova simulator,
* a patient-calibrated physiological simulator,
* a regulatory-grade artificial-pancreas simulator,
* a clinical trial,
* a clinical dosing algorithm, or
* evidence of human safety.

The benchmark does not model every real-world failure mode.

Important limitations include:

* one-day simulation horizons,
* lack of patient-record calibration,
* independent parameter sampling assumptions,
* illness not modeled,
* stress not modeled,
* missed infusion/occlusion faults not modeled,
* prolonged exercise scenarios not comprehensively modeled,
* limited long-term adaptation,
* approximate model-based safety projection.

The source manuscript explicitly characterizes the safety projection as a model-based filter rather than a formal guarantee for human physiology.

---

# Safety Disclaimer

**For research and simulation only.**

Do not:

* connect this software to an insulin pump;
* use the controller to determine insulin doses;
* use the output for medical decisions;
* interpret the safety projection as a clinical safety guarantee; or
* deploy the controller on real patients.

The benchmark evaluates mathematical controllers inside a virtual physiological environment.

---

# Citation

```bibtex
@article{forootan_rsp_admm_pid,
  title   = {RSP-ADMM-PID: Robust Safety-Projected Adaptive Glucose Control Across Heterogeneous Virtual Cohorts},
  author  = {
    Forootan, Ehsan and
    Mahmoudi, Shervin and
    Pashaei, Mohammad and
    Khodadadi Aski, Farbod and
    Bahrami, Fariba and
    Masoudi-Nejad, Ali
  },
  year    = {2026},
  note    = {RSP-ADMM-PID virtual-cohort benchmark}
}
```

---

# Authors

**Ehsan Forootan** — University of Tehran, Department of Electrical and Computer Engineering

**Shervin Mahmoudi** — University of Tehran, Department of Electrical and Computer Engineering

**Mohammad Pashaei** — University of Tehran, Department of Engineering Science

**Farbod Khodadadi Aski** — University of Tehran, Department of Electrical and Computer Engineering

**Fariba Bahrami** — University of Tehran

**Ali Masoudi-Nejad** — University of Tehran, Laboratory of Systems Biology and Bioinformatics

---

# License

The source code is released under the **MIT License**.

Third-party libraries, pretrained components, and referenced materials remain subject to their respective licenses.
