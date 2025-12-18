import numpy as np
import matplotlib.pyplot as plt

# Fidelity range
fidelity = np.linspace(0, 1, 1000)

# Reward functions
def reward_log_squared(f):
    infidelity = np.maximum(1.0 - f, 1e-3)
    return ((f**2) * (-np.log10(infidelity)))


def reward_quadratic(f):
    return f**2


def reward_power(f):
    return f**50


# Evaluate rewards
rewards = {
    "log_squared": reward_log_squared(fidelity),
    "quadratic": reward_quadratic(fidelity),
    "power": reward_power(fidelity),
}

# Plot
plt.figure()
for name, values in rewards.items():
    plt.plot(fidelity, values, label=name)

plt.xlabel("Fidelity")
plt.ylabel("Reward")
plt.title("Comparison of Reward Functions vs Fidelity")
plt.legend()
plt.show()
