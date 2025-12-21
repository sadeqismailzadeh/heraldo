## Attribution & References

This package is an independent implementation and extension of the reinforcement learning protocols for quantum state generation described in:

> **Machine learning for efficient generation of universal photonic quantum computing resources**  
> Amanuel Anteneh, Léandre Brunel, and Olivier Pfister  
> *Optica Quantum*, Vol. 2, No. 4, pp. 296-302 (2024)  
> [DOI: 10.1364/OPTICAQ.523445](https://doi.org/10.1364/OPTICAQ.523445)


The original authors' reference implementation can be found at [Gaussian-Catalysis-Cat-States](https://github.com/YourLinkToTheirRepo).


### Key Optimizations & Differences
While structurally based on the physics presented in the paper, `quantum-agent` introduces several algorithmic and architectural optimizations to improve training speed and scalability:

1.  **Pure State Simulation (vs. Density Matrices):**
    Unlike the original implementation which simulates Density Matrices, this package utilizes **Pure State (Ket) vectors** combined with a quantum trajectory approach (Monitored Loss) for dissipative channels. This reduces the state space representation complexity significantly, allowing for much higher photon cutoff dimensions.

2.  **JIT-Compiled Backend:**
    Critical optical gates (Beamsplitters, Loss Channels, and Fock Measurements) have been patched with **Numba JIT compilation**. This bypasses the standard Strawberry Fields overhead, providing fast speeds for the repeated circuit executions required by RL.

3.  **Memory Management:**
    Includes custom patches to disable operation caching in the Fock backend, preventing memory leaks during long-running training sessions with millions of steps.

4.  **Modern RL Architecture:**
    Built on `gymnasium` and `stable-baselines3`, utilizing a modular `CircuitContext` design that decouples physical parameters from the learning agent.
	
	
## Citing This Work

@misc{quantumagent2025,
  author = {Lastname, Firstname},
  title = {quantum-agent: High-Performance RL for Photonic Quantum Computing},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/repo}}
}