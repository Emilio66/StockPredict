import matplotlib.pyplot as plt
import numpy as np


plt.subplot(131)
x1 = [1,2,3,4,5,6,7]
y1 = [5,12,9,18,13,20,17]
labels = ['T0', 'P1', 'T1', 'P2', 'T2', 'P3', 'T3']
np_x1 = np.array(x1)
np_y1 = np.array(y1)
plt.plot(x1, y1, 'bo')
plt.plot(x1, y1)
plt.title('Price movement in UP trend')
plt.ylim(0, 25)
acc = 0
for i, j in zip(np_x1, np_y1):
	if acc % 2 == 0:
		plt.text(i - 0.3, j - 1.2, labels[acc])
	else:
		plt.text(i, j + 0.5, labels[acc])
	acc += 1
plt.xlabel('Week')
plt.ylabel('Price')


plt.subplot(132)
x2 = [1,2,3,4,5,6,7]
y2 = [18,20,11,15,9,12,5]
labels = ['T0', 'P1', 'T1', 'P2', 'T2', 'P3', 'T3']
np_x2 = np.array(x2)
np_y2 = np.array(y2)
plt.plot(x2, y2, 'bo')
plt.plot(x2, y2)
plt.title('Price movement in DOWN trend')
plt.ylim(0, 25)
acc = 0
for i, j in zip(np_x2, np_y2):
	if acc % 2 == 0:
		plt.text(i - 0.3, j - 1.2, labels[acc])
	else:
		plt.text(i, j + 0.5, labels[acc])
	acc += 1
plt.xlabel('Week')
plt.ylabel('Price')

plt.subplot(133)
x3 = [1,2,3,4,5,6,7]
y3 = [15,20,13,19,14,21,13]
labels = ['T0', 'P1', 'T1', 'P2', 'T2', 'P3', 'T3']
np_x1 = np.array(x3)
np_y1 = np.array(y3)
plt.plot(x3, y3, 'bo')
plt.plot(x3, y3)
plt.title('Price movement in NO trend')
plt.ylim(0, 25)
acc = 0
for i, j in zip(np_x1, np_y1):
	if acc % 2 == 0:
		plt.text(i - 0.3, j - 1.2, labels[acc])
	else:
		plt.text(i, j + 0.5, labels[acc])
	acc += 1
plt.xlabel('Week')
plt.ylabel('Price')


plt.show()
