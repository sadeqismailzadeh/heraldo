import numpy as np
import matplotlib.pyplot as plt

def softplus_reward(f, threshold, alpha):
    # Numerically stable softplus: log(1 + exp(x)) / alpha
    x = alpha * (f - threshold)
    return np.logaddexp(0, x) / alpha

def sigmoid_success(f, threshold, alpha=100):
    # The sigmoid currently in your code
    return 1 / (1 + np.exp(-alpha * (f - threshold)))

# Parameters
threshold = 0.99
fidelities = np.linspace(0.85, 1.0, 500)  # Focus on the high-fidelity region
alphas = [20, 50, 100, 200]

plt.figure(figsize=(10, 6))

# Plot different Alpha values
for a in alphas:
    reward = softplus_reward(fidelities, threshold, a)
    plt.plot(fidelities, reward, label=f'Softplus (alpha={a})', linewidth=2)

# Plot the standard Linear Fidelity for comparison
# Scaled to be visible on the same plot
plt.plot(fidelities, (fidelities - 0.85), '--', color='gray', label='Linear Trend (Reference)', alpha=0.5)

# Plot the Sigmoid for comparison
plt.plot(fidelities, sigmoid_success(fidelities, threshold, 100) * 0.05, 
         ':', color='red', label='Sigmoid/Success (Scaled)', alpha=0.8)

# Formatting
plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.3, label='Threshold (0.99)')
plt.title(f'Softplus Reward vs. Fidelity (Threshold={threshold})', fontsize=14)
plt.xlabel('Fidelity (F)', fontsize=12)
plt.ylabel('Reward Value', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend()

# Annotation to explain the "Trap"
plt.annotate('Below threshold:\nSmall gradient\n(No "Trap")', 
             xy=(0.92, 0.005), xytext=(0.87, 0.02),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.annotate('Above threshold:\nLinear reward\n(No "Cheating")', 
             xy=(0.995, 0.01), xytext=(0.95, 0.04),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.show()