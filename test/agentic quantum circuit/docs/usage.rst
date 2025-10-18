Usage Guide
===========

This guide provides step-by-step instructions on how to set up the environment,
train a new model, and evaluate its performance.

1. Environment Setup
--------------------

First, you need to install the required Python libraries. It is recommended to
use a virtual environment to manage your dependencies.

.. code-block:: bash

   # Create and activate a virtual environment (optional but recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

   # Install the core dependencies
   pip install gymnasium stable-baselines3[extra] strawberryfields numpy scipy tensorflow

2. Training the Agent
---------------------

To train the reinforcement learning agent, run the ``train_quantum_circuit.py``
script. The script will create a ``Train/`` directory where it will save the
TensorBoard logs and model checkpoints.

.. code-block:: bash

   python train_quantum_circuit.py

You can monitor the training progress in real-time by running TensorBoard:

.. code-block:: bash

   python run_tensorboard.py

3. Evaluating a Trained Model
-----------------------------

Once you have a trained model (saved as a ``.zip`` file), you can evaluate its
performance using the ``evaluate_quantum.py`` script.

.. warning::
   The path to the model is hardcoded inside the script. You must open
   ``evaluate_quantum.py`` and edit the ``MODEL_PATH`` variable to point to
   your desired ``.zip`` file before running it.

.. code-block:: bash

   python evaluate_quantum.py

The script will load the agent, run it in the environment, and print the final
fidelity and other performance metrics.

4. Reproducing Figure 3 from the Article
----------------------------------------

The ``article_fig3.py`` script is a specialized evaluation tool designed to
reproduce the results from Figure 3 of the original research paper. It runs the
agent with a specific set of initial states to verify that it has learned the
correct strategy.

.. warning::
   Similar to the standard evaluation script, the path to the model is
   hardcoded. You must open ``article_fig3.py`` and edit the ``MODEL_PATH``
   variable to point to your trained model.

.. code-block:: bash

   python article_fig3.py

This will generate a plot and save it as ``article_fig3_reproduction.png``,
which you can compare to the figure in the paper.
