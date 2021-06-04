import random


class Memory:
    """
    Class of memory for Reinforcement Learning scheme
    """
    def __init__(self, size_max, size_min):
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Adds a single sample to the memory, deleting if necessary
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element

    def get_samples(self, n):
        """
        Returns a batch of samples of size n, or max_size if necessary
        """
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())  # get all the samples
        else:
            return random.sample(self._samples, n)  # get "batch size" number of samples

    def _size_now(self):
        return len(self._samples)
