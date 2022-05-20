import numpy as np
import matplotlib.pyplot as plt


def step(x0, v0, a, dt):
    # Central difference assuming constant step size
    v = v0 + a*dt
    x = x0 + v*dt
    return x, v

def evolve(x0, v0, aa, dt, N):
    xout = np.zeros((N+1,x0.shape[0]))
    vout = np.zeros((N+1,v0.shape[0]))
    x = x0
    v = v0
    xout[0] = x
    vout[0] = v
    for i in range(1,N+1):
        a = aa(i,x,v)
        x, v = step(x,v,a,dt)
        xout[i] = x
        vout[i] = v
    return xout, vout

def roughplot(f, x, y):
    plt.figure(figsize=(30,5))
    step = x.shape[0]//5
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.scatter(x[i*step],y[i*step],c=f[i*step],s=10)
        plt.gca().set_aspect('equal')
        plt.xlim([-15, 20])
        plt.colorbar();