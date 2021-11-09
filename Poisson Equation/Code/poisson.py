import math
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

N_A = 1e22
t_ox = 1e-9
t_si = 5e-7
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0*11.9
epsilon_sio2 = epsilon_0*3.9
delta_psi_MS = 0.21
psi_t = 26e-3
n_i = 1e16
psi_F = psi_t*math.log(N_A/n_i)
q = 1.6e-19


def fun(y, psi):
    A = q*N_A/epsilon_si
    first = psi[1]
    second = -A*(np.exp(-psi[0]/psi_t) - 1 - np.exp(-2*psi_F/psi_t)*(np.exp(psi[0]/psi_t) - 1))
    return np.array([first, second])

def bc(psi_a, psi_b):
    Cox = epsilon_sio2/t_ox
    B = Cox/epsilon_si
    first = +psi_a[1] + B*(Vg - psi_a[0])
    second = psi_b[0]
    return np.array([first, second])

y = np.linspace(0, 5e-7, 500)
psi = np.zeros((2, y.size))

plt.figure(figsize=(10, 8))
Vgs = np.linspace(-2, 2, 100)
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, y, psi, tol=1e-3, max_nodes=20000)
    print(sol.success)
    plt.plot(y, sol.sol(y)[0], label='Vgs=' + str(i)[:4])
# plt.legend(loc='upper right')
plt.xlabel("y")
plt.ylabel("$\psi(y)$")
plt.xlim(0)
plt.show()

psi = np.zeros((2, y.size))
Vgs = np.linspace(-2, 2, 21)
psi_zero = []
for i in Vgs:
    Vg = i
    sol = solve_bvp(fun, bc, y, psi, tol=1e-5, bc_tol=1e-5)
    psi_zero.append(sol.sol(y)[0][0])
plt.plot(Vgs, psi_zero)
plt.xlabel("Vg(V)")
plt.ylabel("Surface potential $\psi(y=0)$")
plt.show()
