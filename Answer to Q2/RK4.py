import numpy as np
import matplotlib.pyplot as plt

k1 = 100/60
k2 = 600/60
k3 = 150/60

E = 1
S = 10
ES = 0
P = 0

step = 0.1
t_final = 30

x_init = np.array([E, S, ES, P], dtype=float)


def f(x):
    e, s, es, p = x
    Vf = k1 * e * s
    Vb = k2 * es
    V3 = k3 * es
    return np.array([-Vf + Vb + V3, -Vf + Vb, Vf - Vb - V3, V3])


def RK4(f, x_init, step, t_final):
    t = 0
    x = x_init
    x_values = [x_init.copy()]
    t_values = [0]
    V_values = [0]
    while True:
        k_1 = f(x)
        k_2 = f(x + step * k_1 / 2)
        k_3 = f(x + step * k_2 / 2)
        k_4 = f(x + step * k_3)
        x += 1/6 * step * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        t += step
        x_values.append(x.copy())
        t_values.append(t)
        V_values.append(k_1[3])
        if t > t_final:
            break
    return x_values, t_values, V_values

x_values, t_values, V_values = RK4(f, x_init, step, t_final)

all_values = [[x[i] for x in x_values] for i in range(4)]

fig = plt.figure()
ax = fig.add_subplot(111)

for values, name in zip(all_values, ("E", "S", "ES", "P")):
    ax.plot(t_values, values, label='{} value'.format(name))
    
ax.set_xlabel('time (seconds)')
ax.set_ylabel(r'Concentration ($\mu$M)')
ax.set_title('Concentration VS Time')
ax.legend()

fig.savefig("concentration_time_plot.png")
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)

SV_values = [(x[1], v) for x, v in zip(x_values, V_values)]
SV_values.sort()

ax.plot([sv[0] for sv in SV_values], [sv[1] for sv in SV_values], label='V')
    
ax.set_xlabel(r'Concentration of S ($\mu$M)')
ax.set_ylabel(r'Rate of change of P ($\mu$M/s)')
ax.set_title('Rate of Change of P VS Concentration of S')
ax.legend()

fig.savefig("VS_plot.png")

print('maximum of V:', max([sv[1] for sv in SV_values]))