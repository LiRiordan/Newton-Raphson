import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
plt.style.use("Solarize_Light2")

epsilon = 0.005
x = 0.7
f = lambda t: 0.1*(t - 5 - 4*(t-1)**3 + (t)**4)
xr = (-1.0,4.0)
yr = (-1.0,14.0)

class Newton_raphson():
    def __init__(self, x: float, func, xr: tuple[float, float], yr: tuple[float, float], epsilon: float):
        """x -> starting value for Newton-Raphson,
        func -> function used for Newton-Raphson,
        epsilon -> stop when |f(X)| < epsilon,
        xr = (x1,x2) -> graphical display from x1 to x2 in x-coords,
        yr = (y1,y2) -> graphical display from y1 to y2 in y-coords."""
        self.x = x
        self.func = func
        self.epsilon = epsilon
        self.xr = xr
        self.yr = yr
        self.NR = np.array(self.NewtonRaphson())
        self.plotty()


    def NewtonRaphson(self, stopping = 100):
        Err = 10000
        L = [[self.x,self.func(self.x)]]
        stop = 0
        t = self.x
        while Err > self.epsilon and stop < stopping:
            X = torch.tensor(t, requires_grad=True)
            Y = self.func(X)
            Y.backward()
            x_prime = X.grad
            t = t - (Y.item()/x_prime.item())
            Err = np.abs(Y.item())
            L.append([t,f(t)])
            stop += 1
        if stop >= stopping:
            print('Bad starting value!')
        return L


    def plotty(self):
        fig = plt.figure()
        ax = plt.axes(xlim = self.xr, ylim = self.yr)
        line, = ax.plot([],[],lw=3)
        def animate(n):
            line.set_xdata(self.NR[:n, 0])
            line.set_ydata(self.NR[:n, 1])
            return line,
        v = np.linspace(-5,5, 200)
        ax.plot (v, f(v), linewidth = 1)
        ani = FuncAnimation(fig, animate, frames=self.NR.shape[0], interval=500)
        plt.show()


NR = Newton_raphson(x,f,xr,yr,epsilon)
