import random
import numpy as np

class ExperienceBuffer(object):
    def __init__(self, max_buffer_size):
        self._buffer = []
        self._max_buffer_size = max_buffer_size

    def __len__(self):
        return len(self._buffer)

    def add(self, experience):
        if len(self) + len(experience) >= self.max_buffer_size:
            self._buffer[0:(len(experience) + len(self)) - self._max_buffer_size] = []
        self._buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self._buffer, size)), [size, 5])
