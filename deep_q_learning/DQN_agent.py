import numpy as np
import random
from collections import deque
import copy


from neural_network.neural_net import NeuralNetwork
from neural_network.layer import Layer
from neural_network.loss_functions import mse, mse_prime
from neural_network.activation_functions import ReLU, Linear


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.memory.append(data)

    def sample(self, batch_size):
        n_samples = min(batch_size, len(self.memory))
        batch = random.sample(self.memory, k=n_samples)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))


    def __len__(self):
        return len(self.memory)

class DQN_agent:
    def __init__(self, action_space, gamma = 0.99, learning_rate = 0.01, epsilon = 0.2, batch_size=50, 
                 loss_function=mse, loss_prime=mse_prime, network=NeuralNetwork()):


        self.q_network = copy.deepcopy(network)
       
        self.action_space = action_space

        self.off_network = copy.deepcopy(self.q_network)
        self.replay_buffer = ReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loss_function = loss_function
        self.loss_prime = loss_prime


    def get_action(self, state):
        if random.random() < self.epsilon:
            chosen_action = random.randrange(self.action_space)
        else:
            state_batch = np.expand_dims(state, axis=0)
            q_values = self.q_network.forward(state_batch)
            chosen_action =  np.argmax(q_values[0])

        return chosen_action


    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        next_q_values = self.off_network.forward(next_states)
        max_next_q = np.amax(next_q_values, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        current_q_predictions = self.q_network.forward(states)
        target_for_network = current_q_predictions.copy()

        batch_indices = np.arange(min(self.batch_size, len(self.replay_buffer)))
        target_for_network[batch_indices, actions] = target_q
        y_pred = current_q_predictions
        y_true = target_for_network

        gradient = self.loss_prime(y_true, y_pred)
        np.clip(gradient, -1, 1, out=gradient)
        self.q_network.backward(gradient, self.learning_rate)

    def update_target_network(self):
        for q_layer_tuple, off_layer_tuple in zip(self.q_network.layers, self.off_network.layers):
            q_layer, _ = q_layer_tuple
            off_layer, _ = off_layer_tuple
            off_layer.weights = np.copy(q_layer.weights)
            off_layer.biases = np.copy(q_layer.biases)

    def learn_total(self, intitial_epsilon=1.0, final_epsilon=0.01, epsilon_decay_rate=0.9999, learning_rate=0.001, target_update_frequency=1000, 
              learning_starts=1000, epochs=2000):
        pass

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("LunarLander-v3", render_mode=None)
    
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.01
    EPSILON_DECAY_RATE = 0.99998
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQUENCY = 1000
    LEARNING_STARTS = 1000
    EPOCHS = 5000
    GAMMA = 0.995

    net = NeuralNetwork()
    net.add(Layer(input_size=8, output_size=64), ReLU())
    net.add(Layer(input_size=64, output_size=64), ReLU())
    net.add(Layer(input_size=64, output_size=4), Linear())

    dqn_agent = DQN_agent(epsilon=INITIAL_EPSILON, gamma=GAMMA, network=net,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    



    total_steps = 0
    
    for i in range(EPOCHS):
        old_observation, _ = env.reset()
        cumulative_reward = 0
        
        while True:
            total_steps += 1
            action = dqn_agent.get_action(old_observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            cumulative_reward += reward
            
            dqn_agent.replay_buffer.push(old_observation, action, reward, new_observation, done)
            old_observation = new_observation

            if len(dqn_agent.replay_buffer) > LEARNING_STARTS:
                dqn_agent.learn()
                
                if dqn_agent.epsilon > FINAL_EPSILON:
                    dqn_agent.epsilon *= EPSILON_DECAY_RATE

            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                dqn_agent.update_target_network()

            if done:
                break
            
        if i % 10 == 0:
            print(f"Epoch: {i}, Reward: {cumulative_reward:.2f}, Epsilon: {dqn_agent.epsilon:.4f}")
       





