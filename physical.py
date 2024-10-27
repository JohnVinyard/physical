from typing import List, Dict, Union
import numpy as np
from scipy.optimize import direct
from soundfile import SoundFile
from io import BytesIO
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
from scipy.signal import stft
from time import time
from numba import jit


# TODO: It might be nice to move this into zounds
def listen_to_sound(
        samples: np.ndarray, 
        wait_for_user_input: bool = True) -> None:
    
    bio = BytesIO()
    with SoundFile(bio, mode='w', samplerate=22050, channels=1, format='WAV', subtype='PCM_16') as sf:
        sf.write(samples.astype(np.float32))
    
    bio.seek(0)
    data = bio.read()
    
    
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    
    if proc.stdin is not None:
        proc.stdin.write(data)
        proc.communicate()
    
    if wait_for_user_input:
        input('Next')
    

def evaluate(recording: np.ndarray):
    """
    Look at time and frequency domain, then listen
    """
    plt.plot(recording[:])
    plt.show()
    
    _, _, spec = stft(recording, 1, window='hann')
    spec = np.flipud(np.abs(spec).astype(np.float32))
    spec = np.log(spec + 1e-3)
    plt.matshow(spec)
    plt.show()
    
    recording = recording / (recording.max() + 1e-3)
    listen_to_sound(recording, True)



class Mass(object):

    def __init__(
            self,
            _id: str,
            position: np.ndarray,
            mass: float,
            damping: float,
            fixed: bool = False):

        super().__init__()
        self._id = _id
        self.position = position.astype(np.float32)
        self.orig_position = self.position.copy()
        self.mass = mass
        self.damping = damping
        self.acceleration = np.zeros_like(self.position)
        self.velocity = np.zeros_like(self.position)
        self.fixed = fixed

    def __str__(self):
        return f'Mass({self._id}, {self.fixed})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self._id)

    def diff(self):
        return self.position - self.orig_position

    def apply_force(self, force: np.ndarray):
        self.acceleration += force / self.mass

    def update_velocity(self):
        self.velocity += self.acceleration

    def update_position(self):
        if self.fixed:
            return

        self.position += self.velocity

    def clear(self):
        self.velocity *= self.damping
        self.acceleration = np.zeros_like(self.acceleration)



class Spring(object):
    def __init__(self, m1: Mass, m2: Mass, tension: float):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.tension = tension

        # 3D vector representing the resting state/length of the spring
        self.m1_resting = self.m1.position - self.m2.position
        self.m2_resting = self.m2.position - self.m1.position

    def __str__(self):
        return f'Spring({self.m1}, {self.m2}, {self.tension})'

    def __repr__(self):
        return self.__str__()

    def masses(self):
        return [self.m1, self.m2]

    def update_forces(self):
        # compute for m1
        current = self.m1.position - self.m2.position
        displacement = self.m1_resting - current
        self.m1.apply_force(displacement * self.tension)

        # compute for m2
        current = self.m2.position - self.m1.position
        displacement = self.m2_resting - current
        self.m2.apply_force(displacement * self.tension)


class SpringMesh(object):
    def __init__(self, springs: List[Spring]):
        super().__init__()
        self.springs = springs

        self.all_masses = set()
        for spring in springs:
            self.all_masses.update(spring.masses())

    def update_forces(self):
        """
        Apply forces exerted by each spring to the connected masses
        """
        for spring in self.springs:
            spring.update_forces()

    def update_velocities(self):
        """
        Update the velocities of each mass based on the accumulated accelerations
        """
        for mass in self.all_masses:
            mass.update_velocity()

    def update_positions(self):
        """
        Update the positions of each mass based on the accumulated velocities
        """
        for mass in self.all_masses:
            mass.update_position()

    def clear(self):
        for mass in self.all_masses:
            mass.clear()


def class_based_plate(n_samples: int, record_all: bool = False):
    mass = 2
    tension = 0.0005
    damping = 0.9998


    width = 8

    boundaries = {0, width - 1}
    masses: List[List[Union[None, Mass]]] = [
        [None for _ in range(width)]
        for _ in range(width)
    ]


    directions = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0,  1],
        [1, -1],  [1, 0],  [1, 1],
    ], dtype=np.int32)

    for x in range(width):
        for y in range(width):

            m = Mass(
                f'{x},{y}',
                np.array([x, y]),
                mass,
                damping,
                fixed=x in boundaries or y in boundaries)

            masses[x][y] = m

    springs: List[Spring] = []

    for x in range(width):
        for y in range(width):
            current = masses[x][y]
            for direction in directions:
                nx, ny = current.position + direction
                nx, ny = int(nx), int(ny)
                try:
                    neighbor = masses[nx][ny]
                    s = Spring(current, neighbor, tension)
                    springs.append(s)
                except IndexError:
                    pass


    mesh = SpringMesh(springs)

    force_target = [2, 2]
    recording_target = [3, 3]

    forces = {
        2048: np.array([10, 10])
    }


    samples = np.zeros((n_samples,))

    for i in range(n_samples):

        f = forces.get(i, None)
        if f is not None:
            print(f'applying force {f} at time step {i}')
            masses[force_target[0]][force_target[1]].apply_force(f)


        # apply the forces exerted by the springs
        mesh.update_forces()

        # update velocities based on the accumulated forces
        mesh.update_velocities()

        # update the positions based upon velocity
        mesh.update_positions()

        # clear the accumulated forces from this iteration and apply
        # damping via friction to the velocity
        mesh.clear()

        if record_all:
            for x in range(width):
                for y in range(width):
                    samples[i] += masses[x][y].diff()[0]

        else:
            samples[i] = masses[recording_target[0]][recording_target[1]].diff()[0]


    return samples



def class_based_spring_mesh(n_samples: int = 1024, record_all: bool = False):
    mass = 3.5
    tension = 0.5
    damping = 0.9998
    n_masses = 50

    force_target = 3
    recording_target = 5

    x_pos = np.linspace(0, 1, num=n_masses)
    positions = np.zeros((n_masses, 3))
    positions[:, 0] = x_pos

    samples = np.zeros((n_samples,))

    forces: Dict[int, np.ndarray] = {
        4096: np.array([100, 0, 0]),
        # 8192: np.array([0.1, 0.01, 0])
    }

    masses = [
        Mass(str(i), pos, mass, damping, fixed=i == 0 or i == n_masses - 1)
        for i, pos in enumerate(positions)
    ]

    springs = [
        Spring(masses[i], masses[i + 1], tension)
        for i in range(n_masses - 1)
    ]

    mesh = SpringMesh(springs)

    for i in range(n_samples):

        f = forces.get(i, None)
        if f is not None:
            print(f'applying force {f} at time step {i}')
            masses[force_target].apply_force(f)


        # apply the forces exerted by the springs
        mesh.update_forces()

        # update velocities based on the accumulated forces
        mesh.update_velocities()

        # update the positions based upon velocity
        mesh.update_positions()

        # clear the accumulated forces from this iteration and apply
        # damping via friction to the velocity
        mesh.clear()

        if record_all:
            for mass in masses:
                samples[i] += mass.diff()[0]
        else:
            samples[i] = masses[recording_target].diff()[0]


    return samples


# def spring_mesh(
#         node_positions: np.ndarray,
#         masses: np.ndarray,
#         tensions: np.ndarray,
#         damping: float,
#         n_samples: int,
#         mixer: np.ndarray,
#         constrained_mask: np.ndarray,
#         forces: Dict[int, np.ndarray]) -> np.ndarray:
#
#     """
#     We assume that the node positions passed in represent the resting length
#     of the springs connecting the nodes
#
#     Args:
#         node_positions (np.ndarray): The N-dimensional starting positions of each
#             mass, (n_masses, dim)
#         masses (np.ndarray): The mass of each node, (n_masses, 1)
#         tensions (np.ndarray): A sparse (n_masses, n_masses, 1) array defining the connectivity and the
#             spring tensions between nodes
#         damping (float): energy dissipation rate
#         n_samples (int): the number of time steps to run the simulation
#         mixer (np.ndarray): The mix over recordings from each node, an (n_masses, 1) tensor,
#             ideally softmax-normalized
#         constrained_mask (np.ndarray): a binary/boolean mask describing which nodes are fixed
#             and immovable
#         forces (Dict[int, np.ndarray]): a sparse representation of where and when forces are applied to
#             the structure, a dict mapping sample -> (n_masses, dim)
#     """
#
#     # TODO: Check to ensure that tensions is symmetrical, or
#     # accept only the upper triangular of the tensions matrix
#     connectivity_mask = tensions > 0
#
#
#     # compute pair-wise distances between each node.  this will
#     # be used to update forces.  This is considered to be the
#     # resting length of each spring.  This is a fully-connected
#     # graph, but only the positions with non-zero tensions
#     # entries are actually connected
#     # TODO: consider using scipy.spatial.distance.pdist
#     dist = np.linalg.norm(
#         node_positions[None, :] - node_positions[:, None],
#         axis=-1,
#         keepdims=True)
#
#     recording: np.ndarray = np.zeros(n_samples)
#
#     # first derivative of node displacement
#     velocities = np.zeros_like(node_positions)
#
#     # second derivative of node displacement
#     accelerations = np.zeros_like(node_positions)
#
#     for t in range(n_samples):
#
#         # determine if any forces were applied at this time step
#         # then, update the forces acting upon each mass
#         # update the positions of each node based on the accumulated forces
#         # finally record from a single dimension of each node's position
#         f = forces.get(t, None)
#         if f is not None:
#             accelerations += f
#
#
#         # compute current pair-wise distances
#         current_dist = np.linalg.norm(
#             node_positions[None, :] - node_positions[:, None],
#             axis=-1,
#             keepdims=True)
#
#         # determine how far each node is from its resting length
#         disp = dist - current_dist
#
#         # we only care about nodes that are actually connected
#         disp *= connectivity_mask
#
#         # the acceleration of each node will be determined by the displacement
#         # of the spring (symmetrical for both connected nodes) * the mass on the
#         # other end of the spring (not symmetrical)
#         accelerations += (tensions * disp) @ (1 / masses)
#
#
#         # update velocities and apply damping
#         velocities += accelerations
#         velocities *= damping
#
#         # update positions for nodes that are not constrained/fixed
#         node_positions += velocities * constrained_mask
#
#         # record the displacement of each node, from its original
#         # position, weighted by mixer
#         recording[t] = mixer @ current_dist
#
#
#     return recording


@jit(nopython=True, nogil=True)
def physical_modelling(
        positions: np.ndarray, 
        mass: float, 
        constrained_mask: np.ndarray,
        connectivity: np.ndarray,
        tension: float, 
        damping: float,
        n_samples: int,
        recording_target_index: int,
        force_application_time: int,
        force_applied: np.ndarray):
    
    """Model the resonance of a physical object as a mesh of nodes connected by
    springs with homogeneous mass, tension and damping

    Args:
        positions (np.ndarray): (n_nodes, 3) array, representing start positions
        mass (float): scalar representing node masses
        accelerations (np.ndarray): (n_nodes, 3) array, representing node accelerations
        constrained_mask (np.ndarray): (n_nodes) boolean array, representing nodes that are fixed
        connectivity (np.ndarray): (n_nodes, n_nodes) binary mask matrix, representing connectivty between nodes
        tension (float): scalar representing mesh tension
        damping (float): scalar representing damping/friction
        force_application_time (int): sample index when force is applied
        force_applied (np.ndarray): (n_nodes, 3) array representing force applied to network 
    """
    
    node_count: int = positions.shape[0]
    dim: int = positions.shape[1]
    
    # acceleration = np.zeros((node_count, dim), dtype=np.float32)

    velocity = np.zeros((node_count, dim), dtype=np.float32)
    recording = np.zeros(n_samples, dtype=np.float32)
    constrained_mask = constrained_mask.reshape((node_count, 1))
    
    # get the number of neighbors for each node to speed up averages in the loop
    neighbor_count = connectivity.sum(axis=-1).reshape((node_count, 1))
    
    for i in range(n_samples):
        
        # update forces first
        
        if i == force_application_time:
            velocity += force_applied
        
        # compute target/home position based averaged current 
        # positions of neighbors
        # KLUDGE: This assumes that all neighbors are equidistant and
        # all tensions are equal
        home_position = (connectivity @ positions) / (neighbor_count + 1e-12)
        direction = home_position - positions
        acceleration = (tension * direction) / mass
        velocity += acceleration
        velocity *= damping

        # update positions
        positions += velocity * constrained_mask
        
        # record from some scalar value of the model
        recording[i] = positions[recording_target_index, 0]
    
    return recording

"""
TODO:
Starting with two matrices


positions: (n_nodes, 3) array, representing start positions
velocities: (n_nodes, 3) array, representing node velocity
masses:    (n_nodes, 1) array, representing node masses
distances: (n_nodes, n_nodes) array, representing distances between nodes
connectivity: (n_nodes, n_nodes) binary mask matrix, representing connectivty between nodes
tensions: (n_nodes, n_nodes) symmetric matrix representing tensions of connections between nodes
external_forces: (n_nodes, 3, time) force applied to each node at time t


At each time step t:
    initialize a (n_nodes, 3) matrix to collect forces
    check if there is an external force applied to this node at this time step
    
    check the current distances between connected nodes and add any resultant
    forces into the temporary matrix .
    
    sum accumulated forces to get each nodes' acceleration
    add accelerations to each node's velocity
    
    add velocities to each nodes' position

At step 1:
    - a force is applied which moves a single node

At step 2:
    - ???
"""

def plate_simulation():
    # TODO: This should be very similar to the string simulation, but with a
    # different adjacency matrix
    pass
    

def string_simulation():
    n_nodes, dim = 64, 3
    
    # nodes are positioned in a line along a single dimension
    positions = np.zeros((n_nodes, dim))
    positions[:, 2] = np.linspace(0, n_nodes, n_nodes)
    
    mass = 1
    tension = 1
    damping = 0.9998
    n_samples = 2**15
    
    n_seconds = n_samples / 22050
    
    # nodes with a zero value won't have position updates
    constrained = np.ones((n_nodes,))
    constrained[[0, -1]] = 0
    
    # create the adjacency matrix.  Since this is a string, each
    # node is connected to the node directly above and below it
    diag = np.ones((n_nodes - 1,))
    adjacency = np.diag(diag, 1) + np.diag(diag, -1)
    # this isn't strictly necessary, but since the first and last node
    # are constrained, we treat them as if they have no neighbors (they
    # are the neighbors of other nodes, however)
    adjacency[0, :] = 0
    adjacency[-1, :] = 0
    
    force_applied = np.zeros((n_nodes, 3))
    force_applied[33, :] = np.array([1, 10, 1])
    
    start = time()
    
    recording = physical_modelling(
        positions=positions,
        mass=mass,
        constrained_mask=constrained,
        connectivity=adjacency,
        tension=tension,
        damping=damping,
        n_samples=n_samples,
        recording_target_index=16,
        force_application_time=1024,
        force_applied=force_applied)
    end = time()
    
    print(f'Generated {n_seconds} seconds of audio in {end - start} seconds')
    evaluate(recording)
    
    
    
    
if __name__ == '__main__':

    # s1 = class_based_spring_mesh(n_samples=2**16, record_all=True)
    # s2 = class_based_spring_mesh(n_samples=2**16, record_all=False)

    # samples = np.concatenate([s1, s2], axis=-1)

    samples = class_based_plate(n_samples=2**15, record_all=False)
    print(samples.shape)
    evaluate(samples)

    # Using numba's @jit is slower the first time, but
    # more than twice as fast on subsequent runs
    # string_simulation()
    # string_simulation()
    # string_simulation()
    
    