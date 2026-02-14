import numpy as np
import matplotlib.pyplot as plt

objective = np.logspace(-6, -2, 500)  # log-spaced x
y = np.log(objective) + 1e3* objective

plt.semilogx(objective, y)
plt.xlabel("objective (log scale)")
plt.ylabel("log(objective) + 1e3·objective")
plt.grid(True, which="both")
plt.show()
