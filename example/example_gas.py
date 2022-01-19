import expdyn
import numpy as np
import matplotlib.pyplot as plt


"""
Generate random data
"""

dx = 1e-3
n_frame = 100

gas = [np.random.uniform(0, 10, (1000, 3))]
for _ in range(n_frame - 1):
    gas.append(gas[-1] + np.random.uniform(-dx, dx, (1000, 3)))


"""
Link positions into trajectories
"""
trajs = expdyn.link(gas, dx=dx, method='trackpy')


"""
Calculate the isf from trajectories
"""
isf, err = expdyn.get_isf_3d(
    trajs,
    q=np.pi*2,  # wavenumber
    length=100,  # maximum lag value,
    sample=20,  # for each time point, the maximum of sample number
)


"""
Plot
"""
tau = np.arange(len(isf))
plt.errorbar(tau, isf, err)
plt.tight_layout()
plt.show()
