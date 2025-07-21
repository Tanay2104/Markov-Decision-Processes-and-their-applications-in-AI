import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sys

from DQN_agent import DQN_agent
from neural_network.layer import Layer
from neural_network.neural_net import NeuralNetwork
from neural_network.activation_functions import ReLU, Linear

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.99997
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 1000
LEARNING_STARTS = 1000
EPOCHS = 3000
GAMMA = 0.995

def train(initial_epsilon = INITIAL_EPSILON, final_epsilon = FINAL_EPSILON, epsilon_decay_rate = EPSILON_DECAY_RATE, learning_rate = LEARNING_RATE,
          batch_size = BATCH_SIZE, target_update_frequency = TARGET_UPDATE_FREQUENCY, learning_starts = LEARNING_STARTS, epochs = EPOCHS, gamma = GAMMA, save=True, visualise=False):
    
    env = gym.make("LunarLander-v3", render_mode=None)
    env.reset()

    action_space=env.action_space.n
    net = NeuralNetwork()
    net.add(Layer(input_size=8, output_size=64), ReLU())
    net.add(Layer(input_size=64, output_size=64), ReLU())
    net.add(Layer(input_size=64, output_size=4), Linear())

    dqn_agent = DQN_agent(epsilon=initial_epsilon, action_space=action_space, gamma=gamma, network=net,
                            learning_rate=learning_rate, batch_size=batch_size)


    total_steps = 0
    cumulative_rewards = []

    print(f"Starting training with following hyperparameters: \n")
    print(f" Learning Rate: {learning_rate}, Epochs: {epochs}, Gamma: {gamma}, Batch Size: {batch_size}, Target update frequency: {target_update_frequency} ")       
    print(f"Initial Epsilon: {initial_epsilon}, Final Epsilon: {final_epsilon}, epsilon decay rate: {epsilon_decay_rate}")
    for i in range(epochs):
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

            if len(dqn_agent.replay_buffer) > learning_starts:
                dqn_agent.learn()
                
                if dqn_agent.epsilon > final_epsilon:
                    dqn_agent.epsilon *= epsilon_decay_rate

            if total_steps % target_update_frequency == 0:
                dqn_agent.update_target_network()

            if done:
                break

        cumulative_rewards.append(cumulative_reward)   
        if i % 10 == 0:
            print(f"Epoch: {i}, Reward: {cumulative_reward:.2f}, Epsilon: {dqn_agent.epsilon:.4f}")

    if save:
        with open('model_logs.txt', 'w') as f:
            f.write(f"This files contains the trained model(s) parameters and hyperparameters")
        with open('model_logs.txt', 'a') as f:
            f.write(f"\nModel hyperparameters\n")
            f.write(f"Learning Rate: {learning_rate}, Epochs: {epochs}, Gamma: {gamma}, Batch Size: {batch_size} ")
            f.write(f"Initial Epsilon: {initial_epsilon}, Final Epsilon: {final_epsilon}, epsilon decay rate: {epsilon_decay_rate}\n")
            for i in range(len(dqn_agent.q_network.layers)):
                layer = dqn_agent.q_network.layers[i]
                f.write(f"Layer {i} weights:\n {layer[0].weights}")
                f.write(f"Layer {i} biases:\n {layer[0].biases}")
                f.write(f"Layer {i} has activation function {layer[1].__class__.__name__}")
            f.write("-----\n")

    env.close()

    if visualise: 
        env = gym.make("LunarLander-v3", render_mode="human")
        env.reset()

        n_samples = 5

        print(f"Simualating {n_samples} episodes...")
        for i in range(n_samples):
            observation, _ = env.reset()
            
            while True:
                action = dqn_agent.get_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    break
        env.close

    return cumulative_rewards

def plot_graph_cumulative_reward(variable, *args):
    for cumulative_reward_list, variable_value in args:
        plt.plot(cumulative_reward_list, label=variable_value)

    plt.xlabel("Number of Epochs")
    plt.ylabel("Cumulative Reward per epsiode")
    plt.title(f"Variation of Cumulative Reward with {variable}")
    plt.legend()
    plt.savefig(f"Results with variable {variable}",  dpi=300, bbox_inches='tight')

def plot_graph_rolling_cumulative_reward(variable, *args):
    rolling_window = 100

    for cumulative_reward_list, variable_value in args:
        cs = np.cumsum(cumulative_reward_list, dtype=float)
        moving_averages = (cs[rolling_window:] - cs[:-rolling_window]) / rolling_window
        
        plt.plot(moving_averages, label=variable_value)
    
    plt.xlabel(f"Number of Epochs (smoothed over {rolling_window} episodes)")
    plt.ylabel("Cumulative Reward per epsiode")
    plt.title(f"Variation of Cumulative Reward with {variable}")
    plt.legend()
    plt.savefig(f"Rolling Results with variable {variable}",  dpi=300, bbox_inches='tight')
    plt.show()


n = 3
if len(sys.argv) < 4:
    print(f"Usage: python lunar_lander.py learning_rate/gamma/batch_size/target_update_frequency visualise[True][False] save_model_logs[True][False]")
    sys.exit()

variable = sys.argv[1]
visualise = (sys.argv[2].lower() == 'true')
save_model_logs = (sys.argv[3].lower() == 'true')
data=[]
if variable=="learning_rate":
    for lr in [0.01, 0.003, 0.002, 0.001, 0.0001]:
        cumulative_rewards = []
        for i in range(n):
            cumulative_rewards.append(train(visualise=visualise, save=save_model_logs, learning_rate=lr))
        cumulative_rewards = np.array(cumulative_rewards)
        data.append((np.average(cumulative_rewards, axis=0), lr))
    #plot_graph_cumulative_reward("Learning Rate", *data)
    plot_graph_rolling_cumulative_reward("Learning Rate", *data)

elif variable=="gamma":
    for gamma in [0.9, 0.99, 0.995, 0.999, 1.000]:
        cumulative_rewards = []
        for i in range(n):
            cumulative_rewards.append(train(visualise=visualise,save=save_model_logs, gamma=gamma))
        cumulative_rewards = np.array(cumulative_rewards)
        data.append((np.average(cumulative_rewards, axis=0), gamma))
    #plot_graph_cumulative_reward("Gamma", *data)
    plot_graph_rolling_cumulative_reward("Gamma", *data)
elif variable=="batch_size":
    for batch_size in [1, 25, 50, 100, 200]:
        cumulative_rewards = []
        for i in range(n):
            cumulative_rewards.append(train(visualise=visualise,save=save_model_logs,batch_size=batch_size))
        cumulative_rewards = np.array(cumulative_rewards)
        data.append((np.average(cumulative_rewards, axis=0), batch_size))
    #plot_graph_cumulative_reward("Batch size", *data)
    plot_graph_rolling_cumulative_reward("Batch size", *data)

elif variable=="target_update_frequency":
    for target_update_frequency in [100, 500, 1000, 4000, 8000]:
        cumulative_rewards = []
        for i in range(n):
            cumulative_rewards.append(train(visualise=visualise,save=save_model_logs,target_update_frequency=target_update_frequency))
        cumulative_rewards = np.array(cumulative_rewards)
        data.append((np.average(cumulative_rewards, axis=0), target_update_frequency))
    #plot_graph_cumulative_reward("Target Update Frequency", *data)
    plot_graph_rolling_cumulative_reward("Target Update Frequency", *data)