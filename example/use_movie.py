import sys
sys.path.insert(0, "..")
import expdyn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


np.random.seed(0)
"""
Generate random data - ideal gas in 3d cubic box
"""

dx = 1e-1
n_frame = 50
n_particle = 20
box = 5

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


"""
Create a Movie with the trajectories
"""
movie = expdyn.Movie(
    trajs,
    blur=None,  # Gaussian blur can be added to smooth the trajectories
    interpolate=False  #  the missing frame can be filled with interpolation
)


"""
Retrieve the particle label at the 1st frames
"""
labels = movie.label(0)
print(labels)


"""
Get the velocities for particle at the 1st frame
"""
velocities = movie.velocity(0)
print(velocities.shape)


"""
Get trajectories between frame 3 and frame 5
"""
trajs_trimmed = movie.get_trimmed_trajs(20, 30)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for t in movie.trajs:
    ax.plot(*t.positions.T, color='silver')

for tt in trajs_trimmed:
    ax.plot(*tt.positions.T, color=cm.rainbow(np.random.random()))
plt.show()


"""
Export movie to XYZ file
"""
movie.save_xyz("movie-gas.xyz")
