import matplotlib.pyplot as plt
import numpy as np


x = np.array([1, 2, 3, 4])
y = np.array([0, 100, 50, 120])
plt.plot(x, y)
plt.text(1 - 0.1, -3, '(1, 0)')
plt.text(2 - 0.15, 101, '(2, 100)')
plt.text(3 - 0.12, 46, '(3, 50)')
plt.hlines(33, 1, 4, colors = "#B0B0B0", linestyles = "dashed")
plt.hlines(50, 1, 4, colors = "#B0B0B0", linestyles = "dashed")
plt.hlines(66, 1, 4, colors = "#B0B0B0", linestyles = "dashed")
plt.hlines(100, 1, 4, colors = "#B0B0B0", linestyles = "dashed")
tx = 3.4
plt.text(tx, 35, '33% retracement')
plt.text(tx, 52, '50% retracement')
plt.text(tx, 68, '66% retracement')

plt.show()