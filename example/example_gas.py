import sys
sys.path.append("..")
import expdyn
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
"""
Generate random data - ideal gas in 3d cubic box
"""

dx = 1e-1
n_frame = 200
n_particle = 2000
box = 100

gas = [  # a list of coordinates in different frames
    np.random.uniform(0, box, (n_particle, 3))
]
for _ in range(n_frame - 1):
    new = gas[-1] + np.random.uniform(-dx, dx, (n_particle, 3))
    gas.append(new)


"""
Link positions into a list of trajectories

Each trajectory is a tulpe (time, position), the time and position
    are numpy arrays
"""
trajs = expdyn.link(gas, dx=dx*1.5, method='trackpy')
trajs = [t for t in trajs if len(t[0]) > 2]


for t in trajs[:5]:
    plt.plot(*t[1].T[:2], marker='+')
plt.show()


"""
Calculate the isf from trajectories. The error is the standard error,
    calcualted as std / sqrt(N_particle)
"""
sigma = 1.0
isf, err = expdyn.get_isf_3d(
    trajs,
    q=np.pi*2 / sigma,  # wavenumber
    length=50,  # maximum lag value,
    sample_num=20,  # for each time point, the maximum of sample number
)


"""
Plot
"""
tau = np.arange(len(isf))
plt.errorbar(tau, isf, err, marker='o', mfc='w')
plt.xlabel("Lag Time / frame")
plt.ylabel("ISF")
plt.tight_layout()
plt.show()
