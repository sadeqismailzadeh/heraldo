.. Agentic Quantum Circuit documentation master file, created by
   sphinx-quickstart on Sat Oct 18 15:22:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Agentic Quantum Circuit's documentation!
===================================================

Project Purpose
---------------

This project uses a reinforcement learning agent to design and optimize a quantum optical circuit. The primary goal is to autonomously generate specific target quantum states, such as squeezed cat states, by controlling the physical parameters of the circuit components.

Problem It Solves
-----------------

Designing quantum circuits is a complex, non-intuitive process that often relies on expert knowledge and extensive trial-and-error. This project automates the design process by framing it as a reinforcement learning problem. An AI agent learns the optimal sequence of operations to achieve a desired quantum state, potentially discovering novel and more efficient circuit configurations than a human designer could.

Key Features
------------

*   **Autonomous Circuit Design:** An AI agent learns to control circuit parameters without human intervention.
*   **Complex State Generation:** Capable of generating highly non-classical states like squeezed cat states.
*   **Simulation-Based:** Uses the Strawberry Fields library to simulate the quantum optical circuit in a Fock basis.
*   **Flexible and Extensible:** The environment can be easily modified to target different quantum states or to incorporate different circuit components.
*   **Performance Monitoring:** Integrated with TensorBoard for real-time monitoring of the agent's learning progress.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`