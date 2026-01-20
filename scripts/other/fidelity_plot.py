import numpy as np
from scipy.special import expit

def plot_function(success_threshold, fidelities):
    """
    Plots the soft success probability function.

    Args:
        success_threshold: The success threshold value.
        fidelities: An array of fidelity values.
        final_probs: An array of final probability values.
        steepness: The steepness parameter (default: 10.0).
    """
    steepness=500.0
    diff = success_threshold - fidelities
    x = np.log(np.maximum(1.0 - diff, 1e-12))
    
    min_infidel=1e-3
    infidelities = np.maximum(1.0 - fidelities, min_infidel)
    log_vals = np.log10(infidelities)  /  np.log10(min_infidel)
    sigmoids = expit(steepness * x ) 
    soft_success_prob = sigmoids

    import matplotlib.pyplot as plt
    plt.plot(fidelities, soft_success_prob)
    plt.xlabel("Fidelities")
    plt.ylabel("Soft Success Probability")
    plt.title("Soft Success Probability Function")
    plt.grid(True)
    plt.show()

# Example Usage (replace with your actual data)
success_threshold = 0.99
fidelities = np.linspace(0.1, 1, 100)
final_probs = np.ones_like(fidelities)  # Example: uniform probabilities

plot_function(success_threshold, fidelities)

