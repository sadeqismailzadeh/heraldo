# Documentation for `thread_manager_callback.py`

## 1. Overview

This script defines a custom callback for the `stable-baselines3` library. A callback is an object that allows you to execute custom code at specific points during the reinforcement learning training loop (e.g., at the start of an episode, after a model update, etc.).

The `ThreadManagerCallback` is a crucial **performance optimization** designed to solve a common bottleneck in parallel RL training. It dynamically manages the number of computational threads used by PyTorch, the underlying deep learning framework for `stable-baselines3`.

## 2. The Problem: CPU Contention in Parallel Training

When using `SubprocVecEnv` (as in `train_quantum_circuit.py`), the training workload is split into two distinct phases:

1.  **Rollout (Data Collection)**: Multiple independent worker processes simulate the environment to collect experience (state, action, reward, next_state tuples). In our setup, each of these processes is configured to be **single-threaded** to avoid interfering with each other.
2.  **Update (Learning)**: The main process gathers the collected experience and uses it to update the neural networks of the PPO agent. This involves heavy matrix multiplications, which can be significantly accelerated by using **multiple CPU cores/threads**.

The problem is that PyTorch's thread setting is global. If we set it to use multiple threads for the update phase, those settings will carry over to the rollout phase, causing the CPU contention issue described in the `train_quantum_circuit.py` documentation.

## 3. The Solution: Dynamic Thread Management

This callback solves the problem by changing the number of PyTorch threads at the exact moments the training loop transitions between the rollout and update phases.

### Core Components

#### `__init__(self, ...)` - Initialization

-   **Purpose**: Sets up the desired thread counts for each phase.
-   **Parameters**:
    -   `rollout_threads`: The number of threads to use during the data collection phase. This should almost always be **1**.
    -   `update_threads`: The number of threads to use during the model update phase. A value of `-1` (the default) intelligently sets this to the total number of available CPU cores (`mp.cpu_count()`).

#### `_on_rollout_start(self)`

-   **Purpose**: This method is automatically called by `stable-baselines3` right *before* it starts collecting data from the parallel environments.
-   **Action**: It executes `torch.set_num_threads(self.rollout_threads)`, forcing PyTorch to use only one thread. This ensures the rollout phase runs efficiently without CPU contention.

#### `_on_rollout_end(self)`

-   **Purpose**: This method is called right *after* the data collection is finished and just *before* the model update begins.
-   **Action**: It executes `torch.set_num_threads(self.update_threads)`, allowing PyTorch to use all available cores to speed up the neural network calculations.

#### `_on_step(self)` - Custom Logging

-   **Purpose**: This method is called after every step in every parallel environment. It provides a convenient hook for custom logging.
-   **Action**: It checks if an episode has just finished (`if done`). If so, it accesses the `info` dictionary, which contains the `max_fidelity` value logged in `QuantumCircuitEnv`. It then uses `self.logger.record(...)` to write this value to the TensorBoard logs under the name `custom/final_fidelity`. This allows for real-time monitoring of the agent's primary performance metric.

## 4. How It's Used

This callback is instantiated in `train_quantum_circuit.py` and passed to the `model.learn()` function as part of a `CallbackList`. `stable-baselines3` then automatically handles calling its methods at the appropriate times.

## 5. Flowchart of Callback Interaction with Training Loop

```mermaid
graph TD
    subgraph Training Loop
        A[Start model.learn()] --> B{Start New Rollout};
        B --> C[Collect Data from Parallel Envs];
        C --> D{Rollout Finished};
        D --> E[Start Model Update];
        E --> F{Update Finished};
        F --> B;
    end

    subgraph Callback Actions
        B -- triggers --> B1[_on_rollout_start];
        B1 --> B2[torch.set_num_threads(1)];

        D -- triggers --> D1[_on_rollout_end];
        D1 --> D2[torch.set_num_threads(MAX_CPUS)];
        
        C -- every step --> C1[_on_step];
        C1 --> C2{Episode Done?};
        C2 -- Yes --> C3[Log final_fidelity to TensorBoard];
    end
```
