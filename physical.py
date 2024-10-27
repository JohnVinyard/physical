from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from soundfile import SoundFile
from io import BytesIO
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
from scipy.signal import stft
from time import time


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



@dataclass
class CompiledSpringMesh:
    positions: np.ndarray
    masses: np.ndarray
    tensions: np.ndarray
    constrained_mask: np.ndarray

    @property
    def n_nodes(self):
        return self.positions.shape[0]

    @property
    def simulation_dim(self):
        return self.positions.shape[-1]

    def force_template(self):
        return np.zeros_like(self.positions)



class SpringMesh(object):
    def __init__(self, springs: List[Spring]):
        super().__init__()
        self.springs = springs

        self.all_masses = set()
        for spring in springs:
            self.all_masses.update(spring.masses())

    def compile(self) -> CompiledSpringMesh:
        return CompiledSpringMesh(
            self.position_array,
            self.mass_array,
            self.tension_array,
            self.constrained_mask)

    @property
    def constrained_mask(self):
        return np.array([0 if m.fixed else 1 for m in self.all_masses], dtype=np.float32)

    @property
    def mass_array(self) -> np.ndarray:
        return np.array([m.mass for m in self.all_masses])

    @property
    def masses(self):
        return list(self.all_masses)

    @property
    def position_array(self) -> np.ndarray:
        return np.array([m.position for m in self.all_masses])

    @property
    def tension_array(self):
        n_masses = len(self.all_masses)
        arr = np.zeros((n_masses, n_masses))
        indexed_masses = {mass: i for i, mass in enumerate(self.all_masses)}

        for spring in self.springs:
            i1 = indexed_masses[spring.m1]
            i2 = indexed_masses[spring.m2]
            arr[i1, i2] = spring.tension
            arr[i2, i1] = spring.tension

        return arr

    @property
    def flat_mixer(self):
        """
        A mixing matrix that records from each node at the same amplitude
        """
        return np.ones((len(self.all_masses),))

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

def build_plate(width: int) -> SpringMesh:
    mass = 2
    tension = 0.005
    damping = 0.9998

    # width = 8

    boundaries = {0, width - 1}
    masses: List[List[Union[None, Mass]]] = [
        [None for _ in range(width)]
        for _ in range(width)
    ]

    directions = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1],
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
    return mesh

def class_based_plate(n_samples: int, record_all: bool = False):
    # mass = 2
    # tension = 0.0005
    # damping = 0.9998
    #
    #
    # width = 8
    #
    # boundaries = {0, width - 1}
    # masses: List[List[Union[None, Mass]]] = [
    #     [None for _ in range(width)]
    #     for _ in range(width)
    # ]
    #
    #
    # directions = np.array([
    #     [-1, -1], [-1, 0], [-1, 1],
    #     [0, -1],           [0,  1],
    #     [1, -1],  [1, 0],  [1, 1],
    # ], dtype=np.int32)
    #
    # for x in range(width):
    #     for y in range(width):
    #
    #         m = Mass(
    #             f'{x},{y}',
    #             np.array([x, y]),
    #             mass,
    #             damping,
    #             fixed=x in boundaries or y in boundaries)
    #
    #         masses[x][y] = m
    #
    # springs: List[Spring] = []
    #
    # for x in range(width):
    #     for y in range(width):
    #         current = masses[x][y]
    #         for direction in directions:
    #             nx, ny = current.position + direction
    #             nx, ny = int(nx), int(ny)
    #             try:
    #                 neighbor = masses[nx][ny]
    #                 s = Spring(current, neighbor, tension)
    #                 springs.append(s)
    #             except IndexError:
    #                 pass
    #
    #
    # mesh = SpringMesh(springs)

    width = 8
    mesh = build_plate(width=width)

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
            mesh.masses[force_target[0]][force_target[1]].apply_force(f)


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


def build_string():
    mass = 3.5
    tension = 0.5
    damping = 0.9998
    n_masses = 50

    x_pos = np.linspace(0, 1, num=n_masses)
    positions = np.zeros((n_masses, 3))
    positions[:, 0] = x_pos

    masses = [
        Mass(str(i), pos, mass, damping, fixed=i == 0 or i == n_masses - 1)
        for i, pos in enumerate(positions)
    ]

    springs = [
        Spring(masses[i], masses[i + 1], tension)
        for i in range(n_masses - 1)
    ]

    mesh = SpringMesh(springs)
    return mesh


def class_based_spring_mesh(
        mesh: SpringMesh,
        force_target: int,
        n_samples: int = 1024):


    samples = np.zeros((n_samples,))

    forces: Dict[int, np.ndarray] = {
        2048: np.array([0.1, 0.1, 0]),
    }

    for i in range(n_samples):

        f = forces.get(i, None)
        if f is not None:
            print(f'applying force {f} at time step {i}')
            mesh.masses[force_target].apply_force(f)


        # apply the forces exerted by the springs
        mesh.update_forces()

        # update velocities based on the accumulated forces
        mesh.update_velocities()

        # update the positions based upon velocity
        mesh.update_positions()

        # clear the accumulated forces from this iteration and apply
        # damping via friction to the velocity
        mesh.clear()

        for mass in mesh.masses:
            samples[i] += mass.diff()[0]


    return samples

def spring_mesh(
        node_positions: np.ndarray,
        masses: np.ndarray,
        tensions: np.ndarray,
        damping: float,
        n_samples: int,
        mixer: np.ndarray,
        constrained_mask: np.ndarray,
        forces: Dict[int, np.ndarray]) -> np.ndarray:

    """
    We assume that the node positions passed in represent the resting length
    of the springs connecting the nodes

    Args:
        node_positions (np.ndarray): The N-dimensional starting positions of each
            mass, (n_masses, dim)
        masses (np.ndarray): The mass of each node, (n_masses, 1)
        tensions (np.ndarray): A sparse (n_masses, n_masses, 1) array defining the connectivity and the
            spring tensions between nodes
        damping (float): energy dissipation rate
        n_samples (int): the number of time steps to run the simulation
        mixer (np.ndarray): The mix over recordings from each node, an (n_masses, 1) tensor,
            ideally softmax-normalized
        constrained_mask (np.ndarray): a binary/boolean mask describing which nodes are fixed
            and immovable.  Positions will be updated via current + (change * constrained), so
            constrained nodes should be equal to 0
        forces (Dict[int, np.ndarray]): a sparse representation of where and when forces are applied to
            the structure, a dict mapping sample -> (n_masses, dim)
    """

    # check that the tension matrix is symmetric, since a single spring with
    # a fixed tension can connect two nodes
    if not np.all(tensions == tensions.T):
        raise ValueError('tensions must be a symmetric matrix')

    orig_positions = node_positions.copy()

    connectivity_mask: np.ndarray = tensions > 0

    # compute vectors representing the resting states of the springs
    resting = node_positions[None, :] - node_positions[:, None]

    # initialize a vector to hold recorded samples from the simulation
    recording: np.ndarray = np.zeros(n_samples)

    # first derivative of node displacement
    velocities = np.zeros_like(node_positions)

    accelerations = np.zeros_like(node_positions)

    for t in range(n_samples):

        # determine if any forces were applied at this time step
        # then, update the forces acting upon each mass
        # update the positions of each node based on the accumulated forces
        # finally record from a single dimension of each node's position
        f = forces.get(t, None)
        if f is not None:
            print(f'applying force {f} at time step {t}')
            accelerations += f

        current = node_positions[None, :] - node_positions[:, None]
        d2 = resting - current
        d1 = -resting + current

        # update m1
        x = (d1 * np.triu(tensions[..., None] * connectivity_mask[..., None])).sum(axis=0)
        accelerations += x / masses[..., None]

        # update m2
        x = (d2 * np.tril(tensions[..., None] * connectivity_mask[..., None])).sum(axis=0)
        accelerations += x / masses[..., None]

        # update velocities and apply damping
        velocities += accelerations

        # update positions for nodes that are not constrained/fixed
        node_positions += velocities * constrained_mask[..., None]

        # record the displacement of each node, from its original
        # position, weighted by mixer
        # TODO: we've already done this above, reuse the node displacement
        # calculation
        disp = node_positions - orig_positions
        mixed = mixer @ disp
        recording[t] = mixed[0]

        # clear all the accumulated forces
        velocities *= damping

        accelerations[:] = 0



    return recording



def optimized_string_simulation(mesh: SpringMesh, force_target: int, n_samples: int = 2**15) -> np.ndarray:
    compiled = mesh.compile()
    force_template = compiled.force_template()

    if force_template.shape[-1] == 3:
        force_template[force_target, :] = np.array([0.1, 0.1, 0])
    elif force_template.shape[-1] == 2:
        force_template[force_target, :] = np.array([0.1, 0.1])

    forces = {
        2048:  force_template,
    }

    samples = spring_mesh(
        compiled.positions,
        compiled.masses,
        compiled.tensions,
        damping=0.9998,
        n_samples=n_samples,
        mixer=mesh.flat_mixer,
        constrained_mask=compiled.constrained_mask,
        forces=forces
    )

    return samples

def compare_class_and_optimized_results(n_samples: int=2**15, samplerate: int = 22050):
    mesh = build_string()
    force_target = 3

    audio_seconds = n_samples / samplerate

    start = time()
    a = class_based_spring_mesh(
        mesh, force_target=force_target, n_samples=n_samples)
    stop = time()
    print(f'class-based spring mesh took {stop - start:.2f} seconds to generate {audio_seconds:.2f} seconds of audio')
    evaluate(a)

    start = time()
    b = optimized_string_simulation(
        mesh, force_target=force_target, n_samples=n_samples)
    stop = time()
    print(f'optimized spring mesh took {stop - start:.2f} seconds to generate {audio_seconds:.2f} seconds of audio')
    evaluate(b)


def check_optimized_plate_sim(
        n_samples: int=2**15,
        width: int = 8,
        force_target:int = 7,
        samplerate: int = 22050):

    audio_seconds = n_samples / samplerate
    mesh = build_plate(width)
    start = time()
    samples = optimized_string_simulation(mesh, force_target=force_target, n_samples=n_samples)
    stop = time()
    print(f'optimized spring mesh took {stop - start:.2f} seconds to generate {audio_seconds:.2f} seconds of audio')
    evaluate(samples)

    
if __name__ == '__main__':

    # compare_class_and_optimized_results(n_samples=2**15)
    check_optimized_plate_sim(n_samples=2**16, width=16, force_target=9)

    # s1 = class_based_spring_mesh(n_samples=2**16, record_all=True)
    # s2 = class_based_spring_mesh(n_samples=2**16, record_all=False)

    # samples = np.concatenate([s1, s2], axis=-1)

    # samples = class_based_plate(n_samples=2**15, record_all=False)
    # print(samples.shape)
    # evaluate(samples)

    # Using numba's @jit is slower the first time, but
    # more than twice as fast on subsequent runs
    # string_simulation()
    # string_simulation()
    # string_simulation()
    
    