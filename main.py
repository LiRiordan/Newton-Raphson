import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
plt.style.use("Solarize_Light2")

epsilon = 0.005
x = 0.7
f = lambda x: 0.1*(x - 5 - 4*(x-1)**3 + (x)**4)

X = torch.tensor(x, requires_grad = True)


def NewtonRaphson(x: float, epsilon: float, f,stopping = 100):
    Err = 10000
    L = [[x,f(x)]]
    stop = 0
    while Err > epsilon and stop < stopping:
        X = torch.tensor(x, requires_grad=True)
        Y = f(X)
        Y.backward()
        x_prime = X.grad
        x = x - (Y.item()/x_prime.item())
        Err = np.abs(Y.item())
        L.append([x,f(x)])
        stop += 1
    if stop >= stopping:
        print('Bad starting value!')
    return L


NR = np.array(NewtonRaphson(x, epsilon, f))


fig = plt.figure()
ax = plt.axes(xlim = (-5,5), ylim = (-5,20))
line, = ax.plot([],[],lw=3)

t = np.linspace(-5,5, 200)
ax.plot (t, f(t), linewidth = 1)

def animate(n):
    line.set_xdata(NR[:n, 0])
    line.set_ydata(NR[:n, 1])
    return line,

ani = FuncAnimation(fig, animate, frames = NR.shape[0], interval = 500)

plt.show()