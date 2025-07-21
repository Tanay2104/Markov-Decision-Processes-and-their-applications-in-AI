import numpy as np
class algo:
    def __init__(self, k):
        self.k = k
        self.c = 2
        self.counter = np.ones(k, dtype=int)
        self.Q_values = np.array([0.0 for i in range(k)], dtype=np.float64)

    def choose(self):
       # A_t = argmax_a [ Q_t(a) + c * sqrt(ln(t) / N_t(a)) ]
       chosen_index = np.argmax(self.Q_values + self.c * np.sqrt(np.log(self.counter.sum())/self.counter))
       self.counter[chosen_index]+=1
       return chosen_index
    
    def update(self, chosen_index, reward):
        self.Q_values[chosen_index]=self.Q_values[chosen_index] + (reward - self.Q_values[chosen_index])/self.counter[chosen_index]
        #Q_new(a) = Q_old(a) + (1/N(a)) * (R - Q_old(a)