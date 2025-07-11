import random
import matplotlib.pyplot as plt
from helpers import flatten

class QLearning():
    def __init__(self, env, alpha = 0.1, epsilon = 0.2, gamma = 0.8):
        self.action_space = flatten.get_all_possible_actions(env.action_space)
        self.env = env
        #self.state_space =  range(env.observation_space.n)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.total_rewards = []

        # self.q_values = {
        #     (state, action): random.random() for state in state_space for action in self.action_space
        # }

        self.q_values = {}

    def improve_q_value_per_episode(self):
        state, info = self.env.reset()
        total_reward = 0
        while True:
            action = self.get_action(state)
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if terminated:
                q_next = 0
                new_action = action 
            else:
                q_values_for_state = {action: self.q_values.get((observation, action), 0.0) for action in self.action_space}
                new_action =  max(q_values_for_state, key=q_values_for_state.get)
                q_next = self.q_values.get((observation, new_action), 0.0)

            current_q = self.q_values.get((state, action), 0.0)

            self.q_values[(state, action)] = current_q + self.alpha * (reward + self.gamma * q_next - current_q)

            state = observation
            total_reward+=reward

            if done:
                break
            
        return total_reward
            
            
    def get_action(self, state):
        if random.random() < self.epsilon:
            chosen_action = random.choice(list(self.action_space))
        else:
            q_values_for_state = {action: self.q_values.get((state, action), 0.0) for action in self.action_space}
            #possible_state_action_pairs = [(actions, self.q_values[(state, actions)]) for actions in self.action_space]
            chosen_action =  max(q_values_for_state, key=q_values_for_state.get)

        return chosen_action

    def train(self, iterations):
        print("Starting training...")
        for i in range(iterations):
            total_reward_per_episode = self.improve_q_value_per_episode()
            self.total_rewards.append(total_reward_per_episode)
            if (i+1)%100 == 0: print(f"Iteration {i+1}...") 
        print("Training Ended")

    def visualise_cumulative_reward_density(self, show=False):
        plt.plot(self.total_rewards)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Cumulative Reward in episode")
        plt.suptitle("Cumulative Reward vs number of episodes plot")
        plt.title(f'Env: {self.env.spec.id}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}', 
                 horizontalalignment='center', fontsize=10, color='gray')
        plt.savefig("Cumulative Reward vs number of episodes plot: Q-Learning")
        if show: plt.show()
        
    def visualise_sample_episode(self):
        import gymnasium as gym
        import time
        env_name = env.spec.id
        env.close()

        env_vis = gym.make(env_name, render_mode='human')
        self.epsilon = 0 

        time.sleep(1)
        print("Starting 3 episodes for visualisation")
        for i in range(3):
            print(f"Episode {i+1}")
            state, info = env_vis.reset()
            done=False
            while not done:
                action = self.get_action(state)
                state, reward, terminated, truncated, info = env_vis.step(action)
                done = terminated or truncated
        env_vis.close()

if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make("Taxi-v3", render_mode=None)

    q_learning_agent = QLearning(
        env = env,
        alpha=0.1,
        epsilon=0.1,
        gamma=0.99
    )


    q_learning_agent.train(iterations=10000)
    q_learning_agent.visualise_cumulative_reward_density(show=False)
    q_learning_agent.visualise_sample_episode()