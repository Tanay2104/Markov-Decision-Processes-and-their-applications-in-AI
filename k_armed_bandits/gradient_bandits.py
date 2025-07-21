import numpy as np
class algo:
    def __init__(self, k):
        self.k = k
        self.alpha = 1
        self.counter=0
        self.H_values = np.zeros(k, dtype=np.float64)
        self.average_reward = 0
        self.probability_distribution = np.array([(np.exp(self.H_values[i])/(np.exp(self.H_values).sum())) for i in range(k)])

    def choose(self):
        # π_t(a) = Pr{A_t = a} = exp(H_t(a)) / Σ_{b=1}^{k} exp(H_t(b))
        chosen_index = np.random.choice(np.arange(self.k), p=self.probability_distribution)
        self.counter+=1
        return chosen_index
    
    def update(self, chosen_index, reward):
        self.average_reward = (self.average_reward*(self.counter-1) + reward)/self.counter
        #H_{t+1}(a) = H_t(a) + α (R_t - R̄_t) (1_{a=A_t} - π_t(a)) if a is the chosen action
        self.H_values[chosen_index]=self.H_values[chosen_index] + self.alpha*(reward - self.average_reward)*(1-self.probability_distribution[chosen_index])
        #H_{t+1}(a) = H_t(a) - α (R_t - R̄_t) π_t(a) if a isn't the chosen action
        for i in range(self.k):
            if i!=chosen_index:
                self.H_values[i] = self.H_values[i] - self.alpha*(reward - self.average_reward)*self.probability_distribution[i]
        self.probability_distribution = np.array([(np.exp(self.H_values[i])/(np.exp(self.H_values).sum())) for i in range(self.k)])