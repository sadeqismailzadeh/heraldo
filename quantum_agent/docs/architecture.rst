Architecture
============

This document provides a high-level overview of the project's structure,
outlining the role of each major component and how they interact.

Core Components
---------------

The project is divided into several key Python files:

*   ``quantum_circuit_env.py``
    This is the heart of the project. It defines the ``QuantumCircuitEnv``, a custom
    `Gymnasium <https://gymnasium.farama.org/>`_ environment that simulates the
    quantum optical circuit using the `Strawberry Fields <https://strawberryfields.ai/>`_
    library. Its responsibilities include:
    - Defining the state (observation) and action spaces.
    - Executing the quantum operations based on the agent's actions.
    - Calculating the fidelity of the resulting state against a target state.
    - Providing rewards to the agent based on this fidelity.

*   ``train_quantum_circuit.py``
    This is the main script for training the reinforcement learning agent. It sets up
    the ``QuantumCircuitEnv``, initializes an RL model (e.g., PPO from
    `Stable Baselines3 <https://stable-baselines3.readthedocs.io/>`_), and runs
    the training loop. It handles logging, model checkpointing, and overall
    training configuration.

*   ``evaluate_quantum.py``
    After a model has been trained and saved, this script is used to evaluate its
    performance. It loads a checkpoint, runs the agent in the environment, and
    records the outcomes, such as the final fidelity achieved and the sequence
    of actions taken.

*   ``article_fig3.py``
    A specialized evaluation script designed to reproduce the results shown in
    Figure 3 of the original research paper. It demonstrates that the trained
    agent has learned the specific strategy required to generate the target state.

Supporting Scripts
------------------

*   ``run_tensorboard.py``
    A utility script to launch the TensorBoard interface, which is used to
    visualize the training progress and analyze the agent's learning curves.

*   ``thread_manager_callback.py``
    A custom callback for the Stable Baselines3 training loop. It allows for
    custom actions to be taken at various points during training, such as
    managing resources or performing complex logging.

*   ``sf_operations_no_cache.py``
    A helper module that patches the Strawberry Fields library to disable
    caching. This is a memory optimization technique that is crucial when

    working with a large Fock-space cutoff dimension.

Workflow and Interaction
------------------------

The typical workflow follows these steps:

1.  **Training:** The user runs ``train_quantum_circuit.py``.
    - The script creates an instance of the ``QuantumCircuitEnv``.
    - The RL agent interacts with the environment for thousands of steps.
    - The agent sends actions to the environment.
    - The environment simulates the circuit and returns the new state (observation) and a reward.
    - The agent updates its policy based on the rewards received.
    - Progress is logged and can be viewed with ``run_tensorboard.py``.
    - Trained models are saved as ``.zip`` files.

2.  **Evaluation:** The user runs ``evaluate_quantum.py``, providing a path to a saved model.
    - The script loads the trained agent.
    - The agent is run in the environment in a deterministic mode.
    - The final state fidelity and other metrics are reported.

Here is a simplified diagram of the interaction during training:

.. code-block:: text

    +----------------------------+      +-------------------------+
    | train_quantum_circuit.py   |      |        User             |
    | (RL Agent / PPO)           |----->| (Starts Training)       |
    +-------------+--------------+      +-------------------------+
                  |
                  |  1. action
                  +---------------------> +-----------------------+
                  |                       | quantum_circuit_env.py|
                  |  2. (obs, reward)     | (Simulation)          |
                  <---------------------+ +----------+------------+
                                                     |
                                                     | Uses
                                                     v
                                        +-----------------------+
                                        |  Strawberry Fields    |
                                        |  (Quantum Engine)     |
                                        +-----------------------+
