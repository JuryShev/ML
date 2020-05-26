import time
import psutil
import matplotlib.pyplot as plt

#%matplotlib notebook
plt.rcParams['animation.html'] = 'jshtml'


fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()

i = 0
x_v, y_v = [], []
j=0

while j<100:
    x_v.append(i)
    y_v.append(psutil.cpu_percent())

    ax.plot(x_v, y_v, color='b')

    fig.canvas.draw()

    ax.set_xlim(left=max(0, i - 100), right=i + 100)

    time.sleep(0.00001)
    i += 1
    fig.show()
    j=j+1

plt.close()
