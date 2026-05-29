from abc import ABC, abstractmethod

import numpy as np
import torch as th


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        """Get a numpy array of weights, one per diffusion step."""

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])

    def weights(self):
        return self._weights


def create_named_schedule_sampler(name, num_timesteps):
    if name in ("uniform", "lossaware"):
        return UniformSampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
