import matplotlib.pyplot as plt
import numpy as np

x = [
	r'$\alpha$',
	r'$w=1$',
	r'$w=t$',
	r'$w= \log \left ( t \right )$',
	r'$w= t^{2}$',
	r'$w= t^{3}$',
	r'$w= t^{4}$',
	r'$w= e^{t}$',
	r'$w= sigm \left ( t \right )$']
y1 = np.array([0.8999, 0.719672128314, 0.785245896851, 0.762295078303, 0.791803274731, 0.704918023016, 0.680543382684, 0.686885240625, 0.678688519314])

plt.plot(x, y1, markersize=12, linewidth=3, label="Train")
plt.plot(x, y1*1.23, markersize=12, linewidth=3, label="Test")

plt.show()
