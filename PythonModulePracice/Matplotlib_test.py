import matplotlib.pyplot as plt
import numpy as np

"""
x = np.linspace(-1,1,50)
y = 2*x + 1
plt.figure()
plt.plot(x,y)
plt.show()
"""

"""
x = np.linspace(-3,3,50)
y1 = 2*x + 1
y2 = x**2
plt.figure(num=3, figsize=(8,5))
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--',label='down')
plt.legend(handles=[l1,l2,],labels=['aaa','bbb'],loc='best')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am X')
plt.ylabel('I am Y')

new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
		[r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0)) # outward, axes
ax.spines['left'].set_position(('data',0)) 
"""

"""
x = np.linspace(-3,3,50)
y = 2*x + 1

plt.figure(num=1, figsize=(8,5))
plt.plot(x,y,)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0, s=50, color='b')
plt.plot([x0,x0],[y0,0],'k--',lw=2.5)

# method 1
####################
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3, rad=.2'))

# method 2
###################
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',fontdict={'size':16,'color':'r'})
"""

"""
n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X) #for color value

plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.scatter(np.arange(5),np.arange(5))

plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(())
plt.yticks(())
"""

plt.show()


