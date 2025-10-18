# Documentation for `train_quantum_circuit.py`

## 1. Overview

This script is the main entry point for training the reinforcement learning (RL) agent. It orchestrates the entire training process, from setting up the environment and the agent to managing the training loop, saving progress, and ensuring computational efficiency.

The agent is trained using the **Proximal Policy Optimization (PPO)** algorithm, a state-of-the-art RL algorithm known for its stability and performance. The implementation is provided by the **`stable-baselines3`** library.

## 2. Critical Configuration: Thread Management

The script begins with a crucial block of code that sets environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) to `'1'`.

-   **Purpose**: To prevent a "thread storm" that cripples performance. When running multiple environments in parallel, each environment is a separate process. If libraries like NumPy, OpenBLAS, or MKL are allowed to use multiple threads within each process, the total number of threads can quickly exceed the number of CPU cores, leading to massive overhead from thread contention and context switching.
-   **Mechanism**: By forcing each process to be single-threaded, we ensure that the parallel environments run efficiently without interfering with each other. This is a critical best practice for parallel RL training in Python.

## 3. Core Components and Workflow

### `main()` Function

The entire logic is wrapped in a `main()` function, which is executed when the script is run directly (`if __name__ == "__main__":`).

#### Step 1: Multiprocessing Setup

-   **Purpose**: To accelerate training by collecting data from multiple environments simultaneously.
-   **Mechanism**:
    1.  It determines the number of available CPU cores and sets `N_ENVS` (the number of parallel environments) accordingly, typically leaving one core free for the main process.
    2.  It uses the `make_vec_env` utility from `stable-baselines3` to create a vectorized environment.
    3.  Crucially, it specifies `vec_env_cls=SubprocVecEnv`. This tells `stable-baselines3` to run each environment in its own separate subprocess, which is essential for true parallelism and avoiding Python's Global Interpreter Lock (GIL).
    4.  `start_method='spawn'` is used for creating these processes, which is the safest and most compatible method across different operating systems (Windows, macOS, Linux).

#### Step 2: Path and Model Configuration

-   It sets up a `Train` directory to store TensorBoard logs and model checkpoints.
-   It defines a consistent naming prefix (`ppo_quantum_circuit`) for all saved files.

#### Step 3: Auto-Resume Logic

-   **Purpose**: To allow training to be stopped and resumed without losing progress. This is essential for long training runs.
-   **Mechanism**:
    1.  The script scans the `Train` directory for any saved checkpoint files (e.g., `ppo_quantum_circuit_80000_steps.zip`).
    2.  It uses a regular expression to find the checkpoint with the highest step number.
    3.  If a checkpoint is found, it is marked as the `latest_checkpoint`, and the corresponding step count is extracted.

#### Step 4: Model Creation or Loading

-   **If `latest_checkpoint` exists**:
    -   The script loads the model from the checkpoint file using `PPO.load(latest_checkpoint, ...)`.
    -   The training will continue from this point.
-   **If no checkpoint is found**:
    -   A new PPO model is created from scratch using `PPO(...)`.
    -   **Hyperparameters**: Key PPO hyperparameters are defined here, such as:
        -   `policy_kwargs`: Defines the neural network architecture for the policy and value functions (e.g., two hidden layers with 256 neurons each).
        -   `learning_rate`: The step size for gradient descent.
        -   `n_steps`: The number of steps each parallel environment runs before the model performs a learning update. The total batch size for an update is `n_steps * N_ENVS`.
        -   `batch_size`: The mini-batch size used within the PPO update.
        -   `gamma`: The discount factor for future rewards.
        -   `ent_coef`: The entropy coefficient, which encourages exploration.
    -   `tensorboard_log=log_dir`: This tells the model to write training statistics to the `Train` directory so they can be visualized with TensorBoard.

#### Step 5: Callbacks

Callbacks are objects that `stable-baselines3` can execute at specific points during training.

-   **`CheckpointCallback`**: Automatically saves the model to a `.zip` file every `save_freq` steps. This is what creates the files used by the auto-resume logic.
-   **`ThreadManagerCallback`**: This is a custom callback (defined in `thread_manager_callback.py`). It dynamically adjusts the number of PyTorch threads:
    -   **During data collection (rollout)**: Sets threads to 1 to align with the single-threaded process configuration.
    -   **During model update (learning)**: Sets threads to the maximum available (`num_cpus`) to speed up the neural network computations, which can be parallelized effectively.
-   **`CallbackList`**: The two callbacks are combined into a list to be passed to the model.

#### Step 6: The Training Loop

-   **`model.learn(...)`**: This is the main function call that starts or resumes the training.
-   **Key Parameters**:
    -   `total_timesteps`: The script calculates the *remaining* number of steps to train to reach a target total (e.g., 7,000,000).
    -   `callback=callback_list`: The combined callbacks are passed here.
    -   `reset_num_timesteps=(False if latest_checkpoint else True)`: This is **CRITICAL** for resuming. Setting it to `False` ensures that the internal step counter of the model continues from where the checkpoint left off, rather than resetting to zero.

#### Step 7: Final Save

-   After the `learn` loop finishes, the final model is saved with a `_final.zip` suffix.

## 4. Flowchart of Script Execution

```mermaid
graph TD
    A[Start Script] --> B{Configure Threading Env Vars};
    B --> C[Setup Parallel Environments (SubprocVecEnv)];
    C --> D{Check for Checkpoints in 'Train' dir};
    D -- Yes --> E[Find Latest Checkpoint];
    D -- No --> F[Create New PPO Model];
    E --> G[Load Model from Checkpoint];
    F --> H[Define Callbacks (Checkpoint, ThreadManager)];
    G --> H;
    H --> I[Calculate Remaining Timesteps];
    I --> J[model.learn(remaining_timesteps, callbacks, reset_num_timesteps=False)];
    J --> K[Training Complete];
    K --> L[Save Final Model];
    L --> M[End];
```
