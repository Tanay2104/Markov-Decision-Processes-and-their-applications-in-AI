# The Gambler's problem is mentioned on page 84, Sutton-Barto RL. This program uses the value iteration algorithm to solve this problem.

import matplotlib.pyplot as plt
import numpy as np

class Gambler:
    def __init__(self, capital=50, target = 100, p_h = 0.4, gamma = 1, threshold = 1e-9):
        self.capital = capital
        self.target = target
        self.p_h = p_h
        self.gamma = gamma
        self.states = np.arange(target+1)
        self.values = np.zeros_like(self.states, dtype=float)
        self.actions = np.arange(1, min(self.capital, self.target - self.capital)+1)
        self.threshold = threshold
        self.policy = np.zeros_like(self.states)

    def get_next_state_reward(self, state, action, heads=True):
        if  heads:
            capital = state + action
        else:
            capital = state - action

        if capital == self.target: reward = 1
        else: reward = 0

        return capital, reward
    
    def iterate_value(self):
        i = 0
        while True:
            delta = 0
            print('Iteration: ', i)
            new_values = np.zeros_like(self.values)
            for state in self.states[1:-1]:
                v = self.values[state]
                self.actions = np.arange(1, min(state, self.target - state)+1)
                action_q_values = np.zeros_like(self.actions, dtype=float)
                for action in self.actions:
                    capital, reward = self.get_next_state_reward(state, action, heads=True)
                    action_q_values[action-1] += self.p_h*(reward + self.gamma*self.values[capital])
                    capital, reward = self.get_next_state_reward(state, action, heads=False)
                    action_q_values[action-1] += (1-self.p_h)*(reward + self.gamma*self.values[capital])
                
               
                new_values[state] = action_q_values.max()
                delta = max(delta, abs(v - new_values[state]))
            
            self.values = new_values
            i+=1
            if delta < self.threshold:
                print('Value function converged')
                break
            elif i > 10000:
                print('Max iterations reached')
                break

    def get_policy(self):
       
        for state in self.states[1:-1]:
            self.actions = np.arange(1, min(state, self.target - state)+1)
            action_q_values = np.zeros_like(self.actions, dtype=float)
          
            for action in self.actions:
                    capital, reward = self.get_next_state_reward(state, action, heads=True)
                    action_q_values[action-1] += self.p_h*(reward + self.gamma*self.values[capital])
                    capital, reward = self.get_next_state_reward(state, action, heads=False)
                    action_q_values[action-1] += (1-self.p_h)*(reward + self.gamma*self.values[capital])
            self.policy[state] = np.argmax(action_q_values)+1
        




    def visualise_result(self):
        width = 14
        height = 7
        plt.figure(figsize=(width, height))

        plt.subplot(1, 2, 1)
        plt.plot(self.values[1:-1])
        plt.xlabel('Capital')
        plt.ylabel('Value Estimates')
        #plt.title(f'The Value Function for p_h = {self.p_h}')

        

        plt.subplot(1, 2, 2)
        plt.plot(self.policy[1:-1])
        plt.xlabel('Capital')
        plt.ylabel('Policy(stakes)')
        #plt.title(f'The Final policy for p_h = {self.p_h}')

        plt.suptitle(f'Solution to Gambler\'s problem for p_h = {self.p_h} and target = {self.target}')

        plt.savefig(f'target={self.target} p_h = {self.p_h}.png', bbox_inches='tight') 

gambler = Gambler(p_h=0.25, target=50)

gambler.iterate_value()
gambler.get_policy()
gambler.visualise_result()