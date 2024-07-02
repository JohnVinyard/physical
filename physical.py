from typing import List
import numpy as np
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
    plt.matshow(np.flipud(np.abs(spec.astype(np.float32))))
    plt.show()
    
    recording = recording / (recording.max() + 1e-3)
    listen_to_sound(recording, True)

 
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
    
    """_summary_

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
    
    acceleration = np.zeros((node_count, dim), dtype=np.float32)
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
    
    # Using numba's @jit is slower the first time, but
    # more than twice as fast on subsequent runs
    string_simulation()
    string_simulation()
    string_simulation()
    