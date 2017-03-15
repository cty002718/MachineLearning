from pylab import *

X = np.linspace(-np.pi, np.pi, 256, endpoint = True)
C,S = np.cos(X), np.sin(X)

figure(figsize=(10,6), dpi=80)
xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
		 [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
yticks([-1,0, +1],
		[r'$-1$', r'$0$', r'$+1$'])
#xmin, xmax = X.min(), X.max()
#ymin, ymax = C.min(), C.max()

#dx = (xmax - xmin) * 0.2
#dy = (ymax - ymin) * 0.2

#xlim(xmin - dx, xmax + dx)
#ylim(ymin - dy, ymax + dy)

plot(X, C, color = "blue", linewidth = 2.0, linestyle = "-", label="cosine")
plot(X, S, color = "red", linewidth = 2.0, linestyle = "-", label="sine")

legend(loc='upper left')
show()

