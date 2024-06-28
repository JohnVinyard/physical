from typing import List
import numpy as np
from soundfile import SoundFile
from io import BytesIO
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
from scipy.signal import stft

# TODO: It might be nice to move this into zounds
def listen_to_sound(samples: np.ndarray, wait_for_user_input: bool = True) -> None:
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


class Node(object):
    
    home: np.ndarray
    pos: np.ndarray
    mass: float
    tension: float
    damping: float
    velocity: np.ndarray
    _acceleration: np.ndarray
    constrained: bool
    neighbors: List['Node']
    
    def __init__(
            self, 
            home: np.ndarray, 
            mass: float, 
            tension: float, 
            damping: float,
            constrained: bool = False,
            neighbors: List['Node'] = []):
        
        super().__init__()
        self.home = home
        self.pos = np.copy(home)
        self.mass = mass
        self.tension = tension
        self.damping = damping
        self.velocity = np.zeros(home.shape)
        self._acceleration = np.zeros(self.velocity.shape)
        self.constrained = constrained
        self.neighbors = neighbors
    
    @property
    def force(self):
        return self.mass * np.linalg.norm(self._acceleration)
    
    def _compute_home_position(self):
        if self.neighbors:
            s = np.stack([n.pos for n in self.neighbors])
            home = np.mean(s, axis=0)
            return home
        
        return self.home
    
    def apply_force(self, force: np.ndarray):
        self.velocity += force
    
    def update(self, iteration: int = 0):
        # compute the direction in which this node will be pulled
        # by gravity or its resting point
        diff = self._compute_home_position() - self.pos
        direction = diff
        
        # acceleration is a function of tension and mass
        acceleration = (self.tension * direction) / self.mass
        self._acceleration = acceleration
        
        # acceleration modifies velocity
        self.velocity += acceleration
        
        # drag reduces velocity
        self.velocity *= self.damping

    def step(self, iteration: int = 0):
        
        if self.constrained:
            # this node is held in place   
            self.pos = self.pos
            return
        
        # velocity modifies position
        self.pos += self.velocity


def plate(
        width: int=8, 
        mass: float=1, 
        tension: float=0.01, 
        damping: float=0.999, 
        target_node_index: List[int] = [0, 0], 
        starting_displacement: np.ndarray = np.ones((3,))):
    
    print('Creating nodes')
    nodes = [
        [Node(
            np.array([0, i, j], dtype=np.float32), 
            mass=mass, 
            tension=tension, 
            damping=damping
        ) for j in range(width)] for i in range(width)]
    
    print('connecting network')
    for i in range(1, width - 1):
        for j in range(1, width - 1):
            n = nodes[i][j]
            n.neighbors = [
                nodes[i - 1][j - 1], 
                nodes[i][j - 1], 
                nodes[i + 1][j - 1], 
                nodes[i + 1][j], 
                nodes[i + 1][j + 1], 
                nodes[i][j + 1], 
                nodes[i - 1][j + 1], 
                nodes[i - 1][j], 
            ]
    
    # print('constrain ends')
    # for i in range(width):
    #     for j in range(width):
    #         if i == 0 or i == width - 1 or j == 0 or j == width - 1:
    #             nodes[i][j].constrained = True
    
    target_node = nodes[target_node_index[0]][target_node_index[1]]
    
    n_samples = 2 ** 15
    recording = np.zeros((n_samples,))
    
    for s in range(n_samples):
        
        for i in range(width):
            for j in range(width):
                node = nodes[i][j]
                if node == target_node and s == 1024:
                    target_node.apply_force(starting_displacement)
                node.update(s)
        
        for i in range(width):
            for j in range(width):
                node = nodes[i][j]
                node.step()
        
        recording[s] = target_node.pos[0]
        
    
    plt.plot(recording[:])
    plt.show()
    
    _, _, spec = stft(recording, 1, window='hann')
    plt.matshow(np.flipud(np.abs(spec.astype(np.float32))))
    plt.show()
    
    recording = recording / (recording.max() + 1e-3)
    listen_to_sound(recording, True)
        
        
        
def multiple_nodes(
        n_nodes: int=10, 
        mass: float=1, 
        tension: float=0.01, 
        damping: float=0.999,
        target_node_index: int=5,
        starting_displacement=np.ones((3,))):

    print(f'creating {n_nodes} nodes')
    
    nodes = [
        Node(
            np.array([0, 0, i], dtype=np.float32), 
            mass=mass, 
            tension=tension, 
            damping=damping) 
        for i in np.linspace(0, 1, n_nodes)
    ]
    
    print(f'connecting network')
    
    # connect the network
    for i in range(1, n_nodes - 1):
        target = nodes[i]
        target.neighbors = [nodes[i - 1], nodes[i + 1]]
        print(f'setting neighbors for node {i}')
        
    print(f'constrain ends')
    # fix the ends
    nodes[0].constrained = True
    nodes[-1].constrained = True
    
    
    print(f'snap all nodes to new homes')
    target_node = nodes[target_node_index]
    
    
    n_samples = 2**15
    
    recording = np.zeros((n_samples,))
    
    for i in range(n_samples):
        for node in nodes:
            if i == 1024 and node == target_node:
                target_node.apply_force(starting_displacement)
            node.update(i)
            
        for node in nodes:
            node.step(i)
        
        recording[i] = target_node.pos[0]
    
    plt.plot(recording[:])
    plt.show()
    
    _, _, spec = stft(recording, 1, window='hann')
    plt.matshow(np.flipud(np.abs(spec.astype(np.float32))))
    plt.show()
    
    recording = recording / (recording.max() + 1e-3)
    listen_to_sound(recording, True)
    

    
if __name__ == '__main__':
    # multiple_nodes(
    #     n_nodes=8, 
    #     mass=16, 
    #     tension=1, 
    #     damping=0.999, 
    #     target_node_index=4, 
    #     starting_displacement=np.array([10, 10, 10]))
    
    plate(
        width=8, 
        mass=16, 
        tension=0.5, 
        damping=0.9998, 
        target_node_index=[1, 3], 
        starting_displacement=np.array([1, 1, 0]))
        
    