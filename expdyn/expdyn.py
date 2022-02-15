import numpy as np
import trackpy as tp
from scipy import ndimage
from scipy.spatial.distance import cdist
from numba import njit
from tqdm import tqdm
from numba.typed import List as nList


@njit
def get_trajectory(labels, frames, target: int):
    time, positions = nList(), nList()
    for t in range(len(labels)):
        index = -1
        label = labels[t]
        frame = frames[t]
        index, found = -1, False

        for i, l in enumerate(label):
            if l == target:
                index, found = i, True
                break

        if index >= 0:
            positions.append(frame[index])  # (xy, xy2, ..., xyn)
            time.append(t)
        elif found:
            return time, positions
    return time, positions


class Trajectory():
    def __init__(self, time, positions, blur=None, velocities=None, blur_velocity=None):
        """
        Args:
            time (np.ndarray): frame number for each positon, dtype=int
            positions (np.ndarray): shape is (n_time, n_dimension)
            blur (float): applying gaussian_filter on each dimension along time axis
            velocities (np.ndarray): velocities at each time points
                                     this is ONLY possible for simulation data
        """
        if len(time) != len(positions):
            raise ValueError("Time points do not match the position number")
        self.time = time
        self.length = len(time)
        if blur:
            self.positions = ndimage.gaussian_filter1d(positions, blur, axis=0)
        else:
            self.positions = positions
        self.p_start = self.positions[0]
        self.p_end = self.positions[-1]
        self.t_start = self.time[0]
        self.t_end = self.time[-1]
        self.velocities = velocities
        if not isinstance(self.velocities, type(None)):
            self.v_end = self.velocities[-1]
        else:
            if blur_velocity:
                positions_smooth = ndimage.gaussian_filter1d(positions, blur_velocity, axis=0)
                self.v_end = (positions_smooth[-1] - positions_smooth[-2]) /\
                             (self.time[-1] - self.time[-2])
            else:
                self.v_end = (self.positions[-1] - self.positions[-2]) /\
                             (self.time[-1] - self.time[-2])

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        return f"trajectory@{id(self):x}"

    def __str__(self):
        return f"trajectory@{id(self):x}"

    def __add__(self, another_traj):
        """
        .. code-block::

            t1 + t2 == t2 + t1 == early + late
        """
        assert type(another_traj) == Trajectory, "Only Fish Trajectories can be added together"
        if self.t_start <= another_traj.t_end:  # self is earlier
            new_time = np.concatenate([self.time, another_traj.time])
            new_positions = np.concatenate([self.positions, another_traj.positions])
            return Trajectory(new_time, new_positions)
        elif self.t_end >= another_traj.t_start:  # self is later
            new_time = np.concatenate([another_traj.time, self.time])
            new_positions = np.concatenate([another_traj.positions, self.positions])
            return Trajectory(new_time, new_positions)
        else:  # there are overlap between time
            return self

    def predict(self, t):
        """
        predict the position of the particle at time t
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.v_end * (t - self.t_end)
        return pos_predict

    def interpolate(self):
        if len(np.unique(np.diff(self.time))) == 0:
            return
        else:
            dimensions = range(self.positions.shape[1])
            pos_nd_interp = []
            for dim in dimensions:
                pos_1d = self.positions[:, dim]
                ti = np.arange(self.time[0], self.time[-1]+1, 1)
                pos_1d_interp = np.interp(x=ti, xp=self.time, fp=pos_1d)
                pos_nd_interp.append(pos_1d_interp)
            self.time = ti
            self.positions = np.vstack(pos_nd_interp).T

    def offset(self, shift):
        """
        offset all time points by an amount of shift
        """
        self.time += shift
        self.t_start += shift
        self.t_end += shift


class Movie:
    """
    Store both the trajectories and positions of experimental data

    .. code-block::

        Movie[f]             - the positions of all particles in frame f
        Movie.velocity(f)    - the velocities of all particles in frame f
        p0, p1 = Movie.indice_pair(f)
        Movie[f][p0] & Movie[f+1][p1] correspond to the same particles

    Attributes:
        trajs (:obj:`list` of :class:`Trajectory`)
        movie (:obj:`dict` of np.ndarray): hold the positions of particle in different frames
        __labels (:obj:`dict` of np.ndarray): hold the ID of particle in different frames
                                              label ``i`` corresponds to ``trajs[i]``
        __indice_pairs (:obj:`dict` of np.ndarray): the paired indices of frame ``i`` and frame ``i+1``

    Example:
        >>> def rand_traj(): return (np.arange(100), np.random.random((100, 3)))  # 3D, 100 frames
        >>> trajs = [rand_traj() for _ in range(5)]  # 5 random trajectories
        >>> movie = Movie(trajs)
        >>> movie[0].shape  # movie[f] = positions of frame f
        (5, 3)
        >>> movie.velocity(0).shape  # movie.velocity(f) = velocities of frame f
        (5, 3)
        >>> pairs = movie.indice_pair(0)  # movie.indice_pairs(f) = (labels in frame f, labels in frame f+1)
        >>> np.allclose(pairs[0], np.arange(5))
        True
        >>> np.allclose(pairs[1], np.arange(5))
        True
        >>> movie[0][pairs[0]].shape  # movie[f][pairs[0]] corresponds to movie[f+1][pairs[1]]
        (5, 3)
    """
    def __init__(self, trajs, blur=None, interpolate=False):
        self.trajs = self.__pre_process(trajs, blur, interpolate)
        self.__sniff()
        self.movie = {}
        self.__velocities = {}
        self.__labels = {}
        self.__indice_pairs = {}

    def __pre_process(self, trajs, blur, interpolate):
        new_trajs = []
        for t in trajs:
            if isinstance(t, Trajectory):
                if blur:
                    new_trajs.append(Trajectory(t.time, t.positions, blur=blur))
                else:
                    new_trajs.append(t)
            elif isinstance(t, dict):
                new_trajs.append(
                    Trajectory(t['time'], t['position'], blur=blur)
                )
            elif type(t) in (tuple, np.ndarray, list):
                new_trajs.append(Trajectory(t[0], t[1], blur=blur))
            else:
                raise TypeError("invalid type for trajectories")
        if interpolate:
            for traj in new_trajs:
                traj.interpolate()
        return new_trajs

    def __sniff(self):
        self.dim = self.trajs[0].positions.shape[1]
        self.max_frame = max([t.time.max() for t in self.trajs])
        self.size = len(self.trajs)

    def __len__(self): return self.max_frame + 1

    def __process_velocities(self, frame):
        """
        Calculate *velocities* at different frames
        if particle ``i`` does not have a position in ``frame+1``, its velocity is ``np.nan``

        Args:
            frame (int): the specific frame to process

        Return:
            tuple: (velocity, (indices_0, indices_1))
                   velocity stores all velocity in ``frame``
        """
        if frame > self.max_frame - 1:
            raise IndexError("frame ", frame, "does not have velocities")
        else:
            position_0 = self[frame]
            position_1 = self[frame + 1]

            velocity = np.empty(position_0.shape)
            velocity.fill(np.nan)

            label_0 = self.__labels[frame]
            label_1 = self.__labels[frame + 1]
            # the order of labels is the same as the order of the positions
            label_intersection = [l for l in label_0 if l in label_1]
            label_intersection, indices_0, indices_1 = np.intersect1d(
                label_0, label_1, assume_unique=True, return_indices=True
            )
            velocity[indices_0] = position_1[indices_1] - position_0[indices_0]
            return velocity, (indices_0, indices_1)

    def __get_positions_single(self, frame):
        if frame > self.max_frame:
            raise StopIteration
        elif frame in self.movie.keys():
            return self.movie[frame]
        else:
            positions = []
            labels = []
            for i, t in enumerate(self.trajs):
                if frame in t.time:
                    time_index = np.nonzero(t.time == frame)[0][0]
                    positions.append(t.positions[time_index])
                    labels.append(i)
            if len(positions) == 0:
                positions = np.empty((0, self.dim))
            else:
                positions = np.array(positions)
            self.movie.update({frame: positions})

            labels = np.array(labels)
            self.__labels.update({frame: labels})
            return positions

    def get_pair(self, f1, f2):
        """
        Return the same individuals in two frames.

        Args:
            f1 (int): the frame index for the first frame
            f2 (int): the frame index for the first frame

        Return:
            (p1, p2): the matched positions in two time points. The positions\
                were stored as numpy arrays. p1[i] and p2[i] referrs to the\
                same identity
        """
        p1 = self[f1]
        p2 = self[f2]
        l1 = self.__labels[f1]
        l2 = self.__labels[f2]
        shared, idx1, idx2 = np.intersect1d(l1, l2, assume_unique=True, return_indices=True)
        return p1[idx1], p2[idx2]

    def __get_velocities_single(self, frame):
        if frame > self.max_frame - 1:
            raise IndexError(f"frame {frame} does not velocities")
        elif frame in self.__velocities.keys():
            return self.__velocities[frame]
        else:
            velocities, indice_pair = self.__process_velocities(frame)
            self.__velocities.update({frame: velocities})
            self.__indice_pairs.update({frame: indice_pair})
            return velocities

    def __get_slice(self, frame_slice, single_method):
        """
        Get the the slice equilivant of single_method
        """
        start = frame_slice.start if frame_slice.start else 0
        stop = frame_slice.stop if frame_slice.stop else self.max_frame + 1
        step = frame_slice.step if frame_slice.step else 1
        for frame in np.arange(start, stop, step):
            yield single_method(frame)

    def __getitem__(self, frame):
        if type(frame) in [int, np.int8, np.int16, np.int32, np.int64]:
            return self.__get_positions_single(frame)
        elif isinstance(frame, slice):
            return self.__get_slice(frame, self.__get_positions_single)
        else:
            raise KeyError(f"can't index/slice Movie with {type(frame)}")

    def add(self, m2):
        """

        Attach another movie to the end of current movie.

        This function should be used in the case where a large
            recording is splited into different movie files.

        Args:
            m2 (Movie): another Movie instance to be attached to the
                end of current movie.

        Return:
            None
        """
        offset = self.max_frame + 1
        for traj in m2.trajs:
            traj.offset(offset)
            self.trajs.append(traj)

        for frame in range(m2.max_frame):
            new_frame = frame + offset
            self.movie.update({
                new_frame: m2[frame]
            })
            self.__labels.update({
                new_frame: m2.label(frame) + self.size
            })
            self.__velocities.update({
                new_frame: m2.velocity(frame)
            })
            self.__indice_pairs.update({
                new_frame: m2.indice_pair(frame)
            })

        self.movie.update({
            m2.max_frame + offset: m2[m2.max_frame]
        })
        self.__labels.update({
            m2.max_frame + offset: m2.label(m2.max_frame) + self.size
        })

        self.max_frame += m2.max_frame + 1
        self.size += len(m2.trajs)

    def velocity(self, frame):
        """
        Retireve velocity at given frame

        Args:
            frame (int / tuple): specifying a frame number or a range of frames

        Return:
            :obj:`list`: velocities of all particle in one frame or many frames,
                         the "shape" is (frame_num, particle_num, dimension)
                         it is not a numpy array because `particle_num` in each frame is different
        """
        if isinstance(frame, int):
            return self.__get_velocities_single(frame)
        elif isinstance(frame, tuple):
            if len(frame) in [2, 3]:
                frame_slice = slice(*frame)
                velocities = list(self.__get_slice(frame_slice, self.__get_velocities_single))
                return velocities
            else:
                raise IndexError(f"Invalid slice {frame}, use (start, stop) or (start, stop, step)")
        else:
            raise KeyError(f"can't index/slice Movie with {type(frame)}, use a Tuple")

    def label(self, frame):
        if frame > self.max_frame:
            return None
        elif frame in self.movie.keys():
            return self.__labels[frame]
        else:
            self[frame]
            return self.__labels[frame]

    def indice_pair(self, frame):
        """
        Return two set of indices, idx_0 & idx_1
        ``Movie[frame][idx_0]`` corresponds to ``Movie[frame + 1][idx_1]``

        Args:
            frame (int): the frame number

        Return:
            :obj:`tuple` of np.ndarray: the indices in ``frame`` and ``frame + 1``
        """
        if frame > self.max_frame - 1:
            raise IndexError(f"frame {frame} does not have a indices pair")
        elif frame in self.__indice_pairs.keys():
            return self.__indice_pairs[frame]
        else:
            velocities, indice_pair = self.__process_velocities(frame)
            self.__velocities.update({frame: velocities})
            self.__indice_pairs.update({frame: indice_pair})
            return indice_pair

    def make(self):
        """
        Go through all frames, making code faster with the object
        """
        for frame in range(self.max_frame):
            self[frame]
            self.velocity(frame)
            self.indice_pair(frame)

    def load(self, filename: str):
        """
        Load a saved file in the hard drive
        """
        with open(filename, 'rb') as f:
            movie = pickle.load(f)
        self.trajs = movie.trajs
        self.movie = movie.movie
        self.__velocities = movie.__velocities
        self.__labels = movie.__labels
        self.__indice_pairs = movie.__indice_pairs
        self.__sniff()

    def save(self, filename: str):
        """
        Save all data using picle
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_trajs(self, blur=0, interpolate=False):
        """
        Reconstruct ``self.trajs``, typically used if :class:`Trajectory` is modified
        """
        new_trajs = []
        for t in self.trajs:
            new_trajs.append(Trajectory(t.time, t.positions, blur=blur))
        if interpolate:
            for traj in new_trajs:
                traj.interpolate()
        self.trajs = new_trajs

    def get_trimmed_trajs(self, t0, t1):
        """
        Get all trajectories that is between frame t0 and t1

        .. code-block::

                     t0                      t1
                      │                      │ 1. Fully accepted
                      │     =========▶       │
              ...................................................
                      │                      │ 2. Trimmed
                      │                      │
                      │                  ====┼──▶
                 ─────┼===▶                  │
                      │                      │
                 ─────┼======================┼───▶
              ...................................................
                      │                      │ 3. Ignored
                      │                      │
              ──────▶ │                      │ ──────▶
            too_eraly │                      │ too late
                      │                      │
                ──────┴──────────────────────┴──────▶ Time

        Args:
            t0 (int): the start frame index.
            t1 (int): the end frame index.

        Return:
            list: the trimmed trajectories between t0 and t1.
        """
        result = []
        for traj in self.trajs:
            too_early = traj.t_end < t0
            too_late  = traj.t_start > t1
            if too_late or too_early:
                continue
            else:
                offset = max(t0 - traj.t_start, 0)
                stop = min(traj.t_end, t1) - traj.t_start
                time = traj.time[offset: stop]
                if len(time) > 1:
                    positions = traj.positions[offset: stop]
                    result.append(Trajectory(time, positions))
        return result

    def save_xyz(self, filename):
        """
        Dump the movie as xyz files. Particle labels indicate the IDs.

        Args:
            filename (str): the name of the xyz file
        """
        if '.xyz' == filename[-4:]:
            fname = filename
        else:
            fname = filename + '.xyz'
        f = open(fname, 'w')
        f.close()

        for i, frame in enumerate(self):
            if len(frame) > 0:
                num, dim = frame.shape
                labels = self.label(i)[:, np.newaxis]
                result = np.concatenate((labels, frame), axis=1)
                with open(fname, 'a') as f:
                    np.savetxt(
                        f, result,
                        delimiter='\t',
                        fmt="\t".join(['%d\t%.8e'] + ['%.8e' for i in range(dim - 1)]),
                        comments='',
                        header='%s\nframe %s' % (num, i)
                    )
            else:
                num, dim = 0, self.dim
                with open(fname, 'a') as f:
                    np.savetxt(
                        f, np.empty((0, self.dim + 1)),
                        delimiter='\t',
                        fmt="\t".join(['%d\t%.8e'] + ['%.8e' for i in range(dim - 1)]),
                        comments='',
                        header='%s\nframe %s' % (0, i)
                    )


class ActiveLinker():
    """
    Link positions into trajectories following 10.1007/s00348-005-0068-7
    Works with n-dimensional data in Euclidean space
    """
    def __init__(self, search_range):
        self.search_range = search_range
        self.labels = None
        self.trajectories = None

    def link(self, frames):
        """
        Getting trajectories from positions in different frames

        Args:
            frames (:obj:`list` of :obj:`numpy.ndarray`): the positions of particles in different frames from the experimental data

        Return:
            :obj:`list`: a collection of trajectories. Each trajectory is represented by a list, [time_points, positions]
        """
        self.labels = self.__get_labels(frames)
        self.trajectories = self.__get_trajectories(self.labels, frames)
        return self.trajectories

    def __predict(self, x1, x0=None, xp=None):
        """
        predict position in frame 2 (x2)
        according to x1, x0, and xp
        """
        if isinstance(x0, type(None)):
            x0 = x1
        if isinstance(xp, type(None)):
            xp = 2 * x0 - x1
        return 3 * x1 - 3 * x0 + xp

    def __get_link_f3(self, f0, f1, f2, links=None):
        if len(f2) == 0:
            return []
        if isinstance(links, type(None)):
            links = []
        new_links = []
        for i0, p0 in enumerate(f0):
            if i0 in [l[0] for l in links]:
                continue
            dist_1 = cdist([self.__predict(p0)], f1)[0]
            candidates_1 = f1[dist_1 < self.search_range]
            labels_1 = np.arange(len(f1))
            labels_1 = labels_1[dist_1 < self.search_range]
            costs = np.empty(labels_1.shape)
            for i, (l1, p1) in enumerate(zip(labels_1, candidates_1)):
                if l1 not in [l[1] for l in links + new_links]:
                    dist_2 = cdist([self.__predict(p1, p0)], f2)[0]
                    costs[i] = np.min(dist_2)
                else:
                    costs[labels_1==l1] = np.inf
            if len(costs) > 0:
                if np.min(costs) < self.search_range * 2:
                    i1 = labels_1[np.argmin(costs)]
                    new_links.append((i0, i1))
        return new_links

    def __get_link_f4(self, fp, f0, f1, f2, old_links):
        if len(f2) == 0:
            return []
        new_links = []
        for ip, i0 in old_links:
            p0 = f0[i0]
            dist_1 = np.linalg.norm(
                    self.__predict(p0, fp[ip])[np.newaxis, :] - f1,  # (n, 3)
                    axis=1
                    )
            candidates_1 = f1[dist_1 < self.search_range]
            if len(candidates_1) == 0:
                continue
            labels_1 = np.arange(len(f1))[dist_1 < self.search_range]
            costs = np.empty(labels_1.shape)
            for i, (l1, p1) in enumerate(zip(labels_1, candidates_1)):
                if l1 not in [l[1] for l in new_links]:  # if l1, p1 is not conflict with other links
                    dist_2 = np.linalg.norm(
                            self.__predict(p1, p0, fp[ip])[np.newaxis, :] - f2,
                            axis=1
                            )
                    costs[i] = np.min(dist_2)
                else:
                    costs[labels_1==l1] = np.inf
            if (len(costs) > 0) and (np.min(costs) < self.search_range * 2):
                i1 = labels_1[np.argmin(costs)]
                new_links.append((i0, i1))
        new_links += self.__get_link_f3(f0, f1, f2, new_links)  # get links start in frame 0
        return new_links

    def __get_links(self, fp, f0, f1, f2, links):
        """
        Get links in two successive frames

        Args:
            fp (:obj:`numpy.ndarray`): previous frame, (dim, n)
            f0 (:obj:`numpy.ndarray`): current frame, (dim, n)
            f1 (:obj:`numpy.ndarray`): next frame, (dim, n)
            f2 (:obj:`numpy.ndarray`): second next frame, (dim, n)
            links (list): index correspondence for particles between fp to f0

        Return:
            list: link from f0 to f1
        """
        if isinstance(fp, type(None)):
            return self.__get_link_f3(f0, f1, f2)
        else:
            return self.__get_link_f4(fp, f0, f1, f2, links)

    def __get_all_links(self, frames):
        """
        Get all possible links between all two successive frames
        """
        frame_num = len(frames)
        links = None
        for n in range(frame_num - 2):
            if n == 0:
                # fp = previous frame
                fp, f0, f1, f2 = None, frames[n], frames[n+1], frames[n+2]
            else:
                fp, f0, f1, f2 = f0, f1, f2, frames[n+2]
            links = self.__get_links(fp, f0, f1, f2, links)
            yield links

    def __get_labels(self, frames):
        links_all = self.__get_all_links(frames)
        labels = [ np.arange(len(frames[0])) ]  # default label in first frame
        label_set = set(labels[0].tolist())
        if len(label_set) > 0:
            new_label = max(label_set) + 1
        else:
            new_label = 0
        for frame, links in zip(frames[1:-1], links_all):
            old_labels = labels[-1]
            new_labels = np.empty(len(frame), dtype=int) # every particle will be labelled
            slots = np.arange(len(frame))
            linked = [l[1] for l in links]
            linked.sort()
            for l in links:
                new_labels[l[1]] = old_labels[l[0]]
            for s in slots:
                if s not in linked:
                    new_labels[s] = new_label
                    label_set.add(new_label)
                    new_label += 1
            labels.append(new_labels)
        return labels

    def __get_trajectories_slow(self, labels, frames):
        frame_nums = len(labels)
        max_value = np.hstack(labels).max()
        trajectories = []
        for i in range(max_value):  # find the trajectory for every label
            traj = {'time': None, 'position': None}
            time, positions = [], []
            for t, (label, frame) in enumerate(zip(labels, frames)):
                if i in label:
                    index = label.tolist().index(i)
                    positions.append(frame[index])  # (xy, xy2, ..., xyn)
                    time.append(t)
            traj['time'] = np.array(time)
            traj['position'] = np.array(positions)
            trajectories.append(traj)
        return trajectories

    @staticmethod
    def __get_trajectories(labels, frames):
        """
        This is used in ActiveLinker

        .. code-block::
            trajectory = [time_points, positions]
            plot3d(*positions, time_points) --> show the trajectory in 3D

        shape of centres: (N x dim)
        """
        max_value = np.hstack(labels).max()
        labels_numba = nList()
        frames_numba = nList()

        for label, frame in zip(labels, frames):
            labels_numba.append(label)
            frames_numba.append(frame)

        trajectories = []
        for target in range(max_value + 1):  # find the trajectory for every label
            result = get_trajectory(labels_numba, frames_numba, target)
            time, positions = result
            trajectories.append(
                (np.array(time), np.array(positions))
            )
        return trajectories


class TrackpyLinker():
    """
    Linking positions into trajectories using Trackpy
    Works with 2D and 3D data. Higher dimensional data not tested.
    """
    def __init__(self, max_movement, memory=0, max_subnet_size=30, **kwargs):
        self.max_movement = max_movement
        self.memory = memory
        self.max_subnet_size = max_subnet_size
        self.kwargs = kwargs

    @staticmethod
    def _check_input(positions, time_points, labels):
        """
        Make sure the input is proper and sequence in time_points are ordered
        """
        assert len(positions) == len(time_points), "Lengths are not consistent"
        if not isinstance(labels, type(None)):
            assert len(positions) == len(labels), "Lengths are not consistent"
            for p, l in zip(positions, labels):
                assert len(p) == len(l), "Labels and positions are not matched"
        time_points = np.array(time_points)
        order_indice = time_points.argsort()
        ordered_time = time_points[order_indice]
        positions = list(positions)
        positions.sort(key=lambda x: order_indice.tolist())
        return positions, ordered_time, labels

    def __get_trajectories(self, link_result, positions, time_points, labels):
        total_labels = []
        for frame in link_result:
            frame_index, link_labels = frame
            total_labels.append(link_labels)

        max_value = np.hstack(total_labels).max()
        labels_numba = nList()
        frames_numba = nList()
        for label, frame in zip(total_labels, positions):
            labels_numba.append(np.array(label))
            frames_numba.append(frame)

        trajectories = []
        for target in tqdm(range(max_value + 1)):  # find the trajectory for every label
            result = get_trajectory(labels_numba, frames_numba, target)
            time, positions = result
            trajectories.append(
                (np.array(time), np.array(positions))
            )
        return trajectories

    def __get_trajectories_slow(self, link_result, positions, time_points, labels):
        """
        this method is slow, will be removed in the future
        """
        trajectories = []
        total_labels = []
        for frame in link_result:
            frame_index, link_labels = frame
            total_labels.append(link_labels)
        for value in tqdm(set(np.hstack(total_labels))):
            traj = self.__get_trajectory(
                value, link_result, positions, time_points, labels
            )
            trajectories.append(traj)
        return trajectories

    def link(self, positions, time_points=None, labels=None):
        """
        Args:
            positions (np.ndarray): shape (time, num, dim)
            time_points (np.ndarray): shape (time, ), time_points may not be continues
            labels (np.ndarray): if given, the result will have a 'label' attribute
                                 which specifies the label values in different frames
                                 [(frame_index, [labels, ... ]), ...], shape (time, num)
        """
        if isinstance(time_points, type(None)):
            time_points = np.arange(len(positions))
        pos, time, labels = self._check_input(positions, time_points, labels)
        tp.linking.Linker.MAX_SUB_NET_SIZE = self.max_subnet_size
        link_result = tp.link_iter(pos, search_range=self.max_movement, memory=self.memory, **self.kwargs)
        trajs = self.__get_trajectories(list(link_result), pos, time, labels)
        return trajs


def __isf_3d(x1, x2, q):
    """
    Calculate the (self) intermediate scattering function (ISF)\
        between two configurations.

    Args:
        x1 (numpy.ndarray): the particle locations, shape (n, 3)
        x2 (numpy.ndarray): the particle locations, shape (n, 3)
        q (float): the wavenumber.

    Return:
        tuple: the value of the self intermediate scattering function, \
            and the standard error
    """
    dist = np.linalg.norm(x2 - x1, axis=1)
    f = np.sinc((q / np.pi) * dist)
    return np.mean(f), np.std(f) / np.sqrt(len(dist))


def __isf_3d_no_drift(x1, x2, q):
    """
    Calculate the (self) intermediate scattering function (ISF)\
        between two configurations. The net transflation between\
        the two configurations were removed.

    Args:
        x1 (numpy.ndarray): the particle locations, shape (n, 3)
        x2 (numpy.ndarray): the particle locations, shape (n, 3)
        q (float): the wavenumber.

    Return:
        tuple: the value of the self intermediate scattering function, \
            and the standard error
    """
    shift = x2 - x1
    shift -= shift.mean(0)[np.newaxis, :]   # remove drift
    dist = np.linalg.norm(shift, axis=1)
    f = np.sinc((q / np.pi) * dist)
    return np.mean(f), np.std(f) / np.sqrt(shift.shape[0])


def link(positions, dx, method='trackpy', **kwargs):
    """
    Link positions into trajectories. Works for N-Dimensional data.

    Args:
        positions (iterable): a collection of locations, each location\
            is a numpy array with shap (N, dimension). The value of N\
            can vary between different time points.
        dx (float): the maxmimu movement for each particle. This parameter\
            was set to reduce the complexity of the linking.
        method (str): trackpy or active. The trackpy method follows the\
            Crocker&Grier's paper in 1996, the active method follows the\
            paper from Ouellette in 2004.
        **kwargs: extra arguments are possible for the trackpy method. See\
            the documentation of trackpy.
    """
    if method == 'trackpy':
        linker = TrackpyLinker(max_movement=dx, **kwargs)
    elif method == 'active':
        linker = ActiveLinker(search_range=dx)
    else:
        raise NotImplementedError(
            f"Linking method {method} has not been implemented"
        )
    return linker.link(positions)


def get_isf_3d(trajectories, q, length=None, sample_num=None, remove_drift=False):
    """
    Calculate the average isf from experimental trajectories

    Args:
        movie (Movie): a instance of movie.
        q (float): the wavenumber.
        legnth (int): the largest lag time of the isf.
        sample_num (int): the maximum number of points sampled per tau value.
        remove_drift (bool): if True, the net translation between \
            configurations will be removed.

    Return:
        (isf, err): the isf is a function of lag time, and the standard\
            error is defined as err = std / sqrt(n)
    """
    traj_objs = [Trajectory(*t) for t in trajectories]
    movie = Movie(traj_objs)
    length_full = len(movie)

    if isinstance(length, type(None)):
        length = length_full

    if isinstance(sample_num, type(None)):
        sample_num = length

    if remove_drift:
        isf_func = __isf_3d_no_drift
    else:
        isf_func = __isf_3d

    isf = np.zeros(length)
    err = np.zeros(length)
    count = np.zeros(length)
    for i in tqdm(range(length_full)):
        for j in range(i+1, min(i + length, length_full)):
            tau = j - i
            if (count[tau] == sample_num):
                continue
            pos_i, pos_j = movie.get_pair(i, j)
            if len(pos_i) < 2:
                continue
            F_mean, F_err = isf_func(pos_i, pos_j, q)
            isf[tau] += F_mean
            err[tau] += F_err
            count[tau] += 1
    count[0] = 1
    isf[0] = 1
    err[0] = np.nan
    return isf / count, err / count
