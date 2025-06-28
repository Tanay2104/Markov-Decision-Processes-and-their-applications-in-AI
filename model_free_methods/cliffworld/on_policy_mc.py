import numpy as np
from cliff import Cliff

# Class for Off policy first visit MC prediction and control, achieved via an epsilon-greedy policy

class MonteCarlo():
    
    def __init__(self, world, epsilon = 0.1, gamma = 0.9):
        self.epsilon = epsilon
        self.world = world
        self.gamma = gamma

        self.q_values = {
            (state, action): np.random.random() for action in self.world.actions for state in self.world.states
        }

        self.returns = {
            (state, action): [] for action in self.world.actions for state in self.world.states
        }

        self.policy = {
           state: np.random.choice(list(self.world.actions)) for state in self.world.states
        }

    def improve_policy(self):
        G = 0
        S_0 = self.world.start
        A_0 = self.policy[S_0]
        t = 1
        while t>0:
            pass
        

