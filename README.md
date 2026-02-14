# Multi-Outcome Circuit Optimization

This repository contains the source code and reproduction scripts for the research article **"Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation"** by S. Ismailzadeh and B. Abedi Ravan.

The package `quantum_agent` implements a framework for optimizing static photonic circuits (Gaussian Boson Sampling-like devices) to generate high-fidelity non-Gaussian states. Unlike traditional single-target optimization, this framework utilizes a **Dynamic Beam Search** strategy to identify and harvest useful quantum states across multiple photon-number-resolving (PNR) measurement patterns simultaneously.

## 📄 Abstract

Photonic quantum computing requires reliable non-Gaussian state generation. Traditional schemes often discard the vast majority of measurement outcomes as "waste." This work proposes a **multi-outcome optimization strategy** that:
1.  **Multiplexes Resources:** Generates diverse resource states (e.g., GKP and Cat states) from a single circuit configuration.
2.  **Harvests Probability:** Aggregates degenerate measurement outcomes to boost the generation rate of specific targets.

We demonstrate this on GKP core states, Schrödinger cat states, binomial codes, and cubic phase states using 2-mode and 3-mode Gaussian circuits.

## 🚀 Key Features

*   **Dynamic Beam Search Optimization:** An algorithm that autonomously discovers high-probability heralding patterns without a priori symmetry assumptions.
*   **Modular Architecture:** Pluggable interfaces for Circuits, Target States, and Reward Mechanisms.
*   **Performance Patches:** Custom JIT-compiled patches for *Strawberry Fields* to accelerate Fock backend operations (Loss channels and Beamsplitters).
*   **Support for Diverse States:** Pre-configured generators for GKP, Cat, Binomial, and Cubic Phase states.

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quantum-agent.git
    cd quantum-agent
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -e .
    ```
    *Note: Primary dependencies include `strawberryfields`, `numpy`, `scipy`, `matplotlib`, `stable-baselines3`, `numba`, and `qutip`.*

## 📂 Repository Structure

*   **`quantum_agent/`**: The core library.
    *   `optimization/`: Contains the `TimeMultiplexedCircuit` interfaces and the **Beam Search Runner** (`time_runner.py`).
    *   `envs/`: Gymnasium environments for RL-based approaches.
    *   `patches/`: Numba-optimized patches for Strawberry Fields operations.
*   **`scripts/`**: Executable scripts for experiments.
    *   `optimize/`: Scripts to run the beam search optimization.
    *   `eval/`: Scripts to visualize Wigner functions and calculate fidelities.
*   **`results/`**: Directory where optimization logs and state files are saved.

## ⚡ Quick Start: Optimization

To run the main multi-outcome optimization algorithm described in Section II of the paper:

```bash
python scripts/optimize/run_time_optimization.py
```

**Configuration:**
Edit the configuration variables at the top of `scripts/optimize/run_time_optimization.py` to change the experiment:
*   `active_target_configs`: Select between GKP, Cat, or Cubic targets.
*   `circuit_config`: Switch between 2-mode or 3-mode gadgets.
*   `OPTIMIZER_METHOD`: Choose between "BASIN" (Basin-Hopping), "CMA" (CMA-ES), etc.

## 📊 Evaluation & Visualization

To analyze the results of an optimization run (plot Wigner functions, check outcome probabilities):

1.  Identify your results folder in `results/` (e.g., `results/opt_Sq3_GKP_...`).
2.  Run the evaluation script:

```bash
python scripts/optimize/eval_time_optimized.py
```

This script automatically finds the latest run, loads the best parameters, and runs a deterministic simulation to generate the output states. It saves the resulting density matrices (`.npy`) and high-resolution plots.

## 🧩 Pre-Computed Results

The archive `paper_results.rar` contains the raw data and optimization logs used to generate the tables and figures in the paper. This includes results not explicitly presented in the final manuscript (e.g., additional hyperparameter sweeps).

## 📚 Citation

<!-- If you use this code in your research, please cite:

```bibtex
@article{ismailzadeh2026multioutcome,
  title={Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation},
  author={Ismailzadeh, S. and Abedi Ravan, B.},
  journal={Department of Physics, IASBS & Shahid Sattari University},
  year={2026}
}
``` -->

## 📝 License

This project is licensed under the MIT License.