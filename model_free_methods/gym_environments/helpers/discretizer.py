import numpy as np
class Discretizer:
    """
    A helper class to discretize a continuous state space.
    """
    def __init__(self, env, num_bins_per_dimension):
        self.env = env
        self.num_bins_per_dimension = num_bins_per_dimension
        
        # Create bins for each dimension of the state space
        self.bins = [
            np.linspace(low, high, num_bins) for low, high, num_bins in zip(
                self.env.observation_space.low,
                self.env.observation_space.high,
                self.num_bins_per_dimension
            )
        ]

    def get_discrete_state(self, continuous_state):
        """
        Takes a continuous state (numpy array) and returns a discrete state (tuple of ints).
        """
        indices = []
        for i in range(len(continuous_state)):
            # np.digitize finds the index of the bin the value belongs to
            index = np.digitize(continuous_state[i], self.bins[i]) - 1
            indices.append(index)
        
        # A tuple of integers is a valid, hashable dictionary key
        return tuple(indices)