from scipy import integrate
import matplotlib.pyplot  as plt
import numpy as np

def model(y,t):

    dy1dt = y[1]
    dy2dt = -0.05*3/2*10*np.sin(y[0]+np.pi)+3/2*(-2)
    dydt = np.array([dy1dt,dy2dt])

    return dydt


# initial condition
y0 = np.array([np.pi/2,0])

# time points
t = np.linspace(0,20)

# solve ODE
y = integrate.odeint(model,y0,t)

plt.plot(t,y[:,0])
plt.show()