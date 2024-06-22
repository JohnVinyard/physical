import numpy as np
# import zounds
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
    
    def __init__(
            self, 
            home: np.ndarray, 
            mass: float, 
            tension: float, 
            damping: float):
        
        super().__init__()
        self.home = home
        self.pos = np.copy(home)
        self.mass = mass
        self.tension = tension
        self.damping = damping
        
        self.velocity = np.zeros(home.shape)
        
        self._acceleration = np.zeros(self.velocity.shape)
    
    @property
    def force(self):
        return self.mass * np.linalg.norm(self._acceleration)
    
    def displace(self, new_pos: np.ndarray) -> None:
        self.pos = new_pos
    
    def step(self, iteration: int = 0):
        diff = self.home - self.pos
        # direction = diff / (np.linalg.norm(diff) + 1e-8)
        direction = diff
        
        acceleration = (self.tension * direction) / self.mass
        self._acceleration = acceleration
        
        self.velocity += acceleration
        
        self.pos += self.velocity
        
        self.velocity *= self.damping
        
        
    
    
if __name__ == '__main__':
    node = Node(
        np.array([0, 0, 0], dtype=np.float32) + 1e-3, 
        mass=500, 
        tension=0.1, 
        damping=0.999)
    
    node.displace(np.array([1, 1, 1], dtype=np.float32) * 1000)
    
    n_samples = 2**15
    
    recording = np.zeros((n_samples,))
    
    for i in range(n_samples):
        node.step(i)
        recording[i] = node.pos[1]
    
    plt.plot(recording[:])
    plt.show()
    
    _, _, spec = stft(recording, 1, window='hann')
    plt.matshow(np.flipud(np.abs(spec.astype(np.float32))))
    plt.show()
    
    recording = recording / (recording.max() + 1e-3)
    listen_to_sound(recording, True)
        
    