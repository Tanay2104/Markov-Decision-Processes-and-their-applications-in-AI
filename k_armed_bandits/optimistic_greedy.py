import numpy as np
class algo:
    def __init__(self, k):
        self.k = k
        self.counter=np.zeros(k, dtype=int)
        self.Q_values = np.array([5.0 for i in range(k)], dtype=np.float64)

    def choose(self):
        max_index = np.argmax(self.Q_values)
        self.counter[max_index]+=1
        return max_index
    
    def update(self, max_index, reward):
        self.Q_values[max_index]=self.Q_values[max_index] + (reward - self.Q_values[max_index])/self.counter[max_index]
        #Q_new(a) = Q_old(a) + (1/N(a)) * (R - Q_old(a)