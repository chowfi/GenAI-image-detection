import collections
import random
import torch
from typing import Callable
from torchdata.datapipes.iter import IterDataPipe


def get_second_entry(sample):
    return sample[1]


class UnderSamplerIterDataPipe(IterDataPipe):
    """Dataset wrapper for under-sampling.

    Copied from: https://github.com/MaxHalford/pytorch-resample/blob/master/pytorch_resample/under.py
    Modified to work with multiple labels.
    
    MIT License

    Copyright (c) 2020 Max Halford

    This method is based on rejection sampling.

    Parameters:
        dataset
        desired_dist: The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values are normalised so that sum up
            to 1.
        label_getter: A function that takes a sample and returns its label.
        seed: Random seed for reproducibility.

    Attributes:
        actual_dist: The counts of the observed sample labels.
        rng: A random number generator instance.

    References:
        - https://www.wikiwand.com/en/Rejection_sampling

    """

    def __init__(self, dataset: IterDataPipe, desired_dist: dict,
                 label_getter: Callable = get_second_entry, seed: int = None):

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}
        self.label_getter = label_getter
        self.seed = seed

        self.actual_dist = collections.Counter()
        self.rng = random.Random(seed)
        self._pivot = None

    def __iter__(self):

        for dp in self.dataset:
            y = self.label_getter(dp)

            self.actual_dist[y] += 1

            # To ease notation
            f = self.desired_dist
            g = self.actual_dist

            # Check if the pivot needs to be changed
            if y != self._pivot:
                self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
            else:
                yield dp
                continue

            # Determine the sampling ratio if the observed label is not the pivot
            M = f[self._pivot] / g[self._pivot]
            ratio = f[y] / (M * g[y])

            if ratio < 1 and self.rng.random() < ratio:
                yield dp

    @classmethod
    def expected_size(cls, n, desired_dist, actual_dist):
        M = max(
            desired_dist.get(k) / actual_dist.get(k)
            for k in set(desired_dist) | set(actual_dist)
        )
        return int(n / M)