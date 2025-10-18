# Documentation for `article_fig3.py`

## 1. Overview

This script is designed for the **quantitative analysis** and **validation** of the trained reinforcement learning agent. Its primary goal is to replicate the results presented in Figure 3 of the original research paper on which this project is based.

To do this, it runs the trained agent over a large number of episodes, collects detailed statistics about its performance, and generates a series of plots that visualize the distribution of these statistics.

## 2. Key Differences from `evaluate_quantum.py`

-   **Purpose**: Quantitative (this script) vs. Qualitative (`evaluate_quantum.py`).
-   **Scale**: Runs many episodes (`NUM_EPISODES_TO_COLLECT = 1250`) vs. a few (`num_episodes = 10`).
-   **Execution**: Uses parallel environments (`SubprocVecEnv`) for fast data collection vs. a single environment.
-   **Output**: Generates statistical plots (histograms) of aggregated data vs. a Wigner function plot for a single episode.

## 3. Core Functionality and Workflow

### Step 1: Configuration

-   **Threading Limits**: Just like `train_quantum_circuit.py`, it starts by setting environment variables to limit NumPy/BLAS threads to 1. This is critical for the performance of the parallel environments.
-   **Simulation Parameters**: Defines constants like `CUTOFF_DIM`, `MAX_STEPS`, etc. These **must match** the parameters used to train the model being evaluated.
-   **Evaluation Parameters**:
    -   `NUM_EPISODES_TO_COLLECT`: The total number of episodes to run to gather statistically significant data.
    -   `N_ENVS`: The number of parallel environments to use for data collection.
-   **Analysis Parameters**: Defines thresholds for analysis, such as what fidelity level is considered a "success."

### Step 2: Setup and Model Loading

-   It creates a vectorized, parallel environment using `make_vec_env` and `SubprocVecEnv`, identical to the setup in the training script.
-   It loads the trained PPO model from the `ppo_quantum_circuit.zip` file.

### Step 3: Parallel Evaluation and Data Collection

-   This is the main data-gathering loop. It runs until the desired number of episodes has been collected.
-   **Data Structures**: It initializes lists to store the results from completed episodes (`fidelities`, `episode_lengths`, etc.) and NumPy arrays to track the state of ongoing episodes in each parallel environment (`current_episode_steps`, etc.).
-   **The Loop**:
    1.  `model.predict(...)` gets actions for all `N_ENVS` environments at once.
    2.  `env.step(...)` executes these actions in the parallel environments.
    3.  It then iterates through the results for each environment (`for i in range(N_ENVS)`).
    4.  It updates trackers for the current step count and total measured photons.
    5.  **Detecting Resets**: It includes logic to detect when the agent performs a "reset" action (by making the beam splitter fully reflective). This is a specific behavior learned by the agent in the paper's strategy, and this script tracks the number of steps between these resets.
    6.  **Handling Done Episodes**: When an episode in one of the environments is finished (`if dones[i]`), it appends the final statistics (fidelity, total steps, photons) to the data lists and resets the trackers for that specific environment.
-   A `tqdm` progress bar is used to provide a visual indication of the data collection progress.

### Step 4: Analysis and Plotting

-   Once the data collection is complete, the `env.close()` method properly shuts down all the subprocesses.
-   **Calculate Statistics**: It converts the data lists to NumPy arrays and calculates summary statistics, such as the average final fidelity and the success rate (the percentage of episodes exceeding a fidelity threshold).
-   **Generate Plots**: It calls the `plot_results` function, which uses `matplotlib` to create a 2x2 grid of histograms. Each subplot corresponds to one of the panels in the paper's Figure 3:
    -   **(a) Output State Fidelity**: Distribution of final fidelities.
    -   **(b) Total Detected Photons**: Distribution of the total number of photons measured per successful segment.
    -   **(c) Steps per Episode**: Distribution of the total episode lengths.
    -   **(d) Steps between Resets**: Distribution of the number of steps the agent takes before performing a reset action.

## 4. Flowchart of Script Execution

```mermaid
graph TD
    A[Start Script] --> B{Configure Threading Env Vars};
    B --> C[Setup Parallel Environments (SubprocVecEnv)];
    C --> D[Load Trained PPO Model];
    D --> E[Initialize Data Lists and Trackers];
    E --> F[Start Collection Loop (while episodes < total)];
    F --> G[Predict actions for all envs];
    G --> H[Step all envs in parallel];
    H --> I[Loop through results of each env];
    I --> J{Episode in this env done?};
    J -- Yes --> K[Append final data (fidelity, steps, photons) to lists];
    K --> L[Reset trackers for this env];
    J -- No --> M[Update trackers (step count, photons)];
    L --> M;
    M --> I;
    I -- All envs processed --> F;
    F -- Loop finished --> N[Close Environments];
    N --> O[Calculate Summary Statistics (Avg Fidelity, Success Rate)];
    O --> P[Generate 2x2 Matplotlib Histograms];
    P --> Q[Show Plot];
    Q --> R[End];
```
