import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(8)
y = np.array([0.719672128314, 0.785245896851, 0.762295078303, 0.791803274731, 0.704918023016, 0.680543382684, 0.686885240625, 0.678688519314])
w=0.2
opacity = 0.4
plt.bar(x-w, y,width=0.2,alpha=opacity,color='b',align='center')
plt.bar(x, y*1.2,width=0.2,alpha=opacity,color='g',align='center')
plt.bar(x+w+w, y*1.22,width=0.2,alpha=opacity,color='y',align='center')
plt.bar(x+w, y*1.42,width=0.2,alpha=opacity,color='c',align='center')

#plt.ylim(0.62, 0.82)
#ax.set_yticks(np.linspace(0.62,0.82,11))
ax.set_xticklabels( (r'$\alpha$',
	r'$w=1$',
	r'$w=t$',
	r'$w= \log \left ( t \right )$',
	r'$w= t^{2}$',
	r'$w= t^{3}$',
	r'$w= t^{4}$',
	r'$w= e^{t}$',
	r'$w= sigm \left ( t \right )$') ,rotation=20)


plt.show()
