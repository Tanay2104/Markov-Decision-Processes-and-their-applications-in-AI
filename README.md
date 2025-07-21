# Summer of Science 2025: Markov-Decision-Processes-and-their-applications-in-AI
**Author:** Tanay Jha

## Project Overview
This project presents a comprehensive study of Reinforcement Learning (RL), tracing a path from the foundational principles of automata and logic to the practical implementation of a Deep Q-Network (DQN) from scratch. The work is structured as a series of self-contained modules, each tackling a core concept in RL with both theoretical explanations (see `/reports`) and practical code implementations.

The project demonstrates a systematic progression of learning:
1.  **Exploration vs. Exploitation:** Understanding the core trade-off using classic Multi-Armed Bandit problems.
2.  **Model-Based Planning:** Solving environments with known dynamics using Value Iteration and Policy Iteration.
3.  **Model-Free Learning:** Developing agents that learn directly from interaction in environments with unknown dynamics, using Monte Carlo and Temporal-Difference methods (Q-Learning, SARSA).
4.  **Deep Reinforcement Learning:** Scaling up to complex, high-dimensional state spaces by implementing a DQN from scratch to solve the `LunarLander-v3` environment.

---

## Project Structure
The repository is organized into four main parts, reflecting a logical progression of topics:

- **`/reports`**: Contains the initial Plan of Action and the final, detailed project report.
- **`/k_armed_bandits`**: Explores the exploration-exploitation dilemma using various bandit algorithms.
- **`/dynamic_programming`**: Implements model-based RL algorithms (Value and Policy Iteration) to solve classic problems.
- **`/model_free_methods`**: Implements model-free algorithms (Monte Carlo, Q-Learning, SARSA) for environments where the model is unknown.
- **`/deep_q_learning`**: Implements a DQN agent and a custom neural network library to solve the `LunarLander-v3` environment.

---

## How to Run Experiments
### Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install numpy matplotlib gymnasium pygame
#Pygame is required for visualisation of gym environments
```
### Running the code
To run the code for each module, navigate to its directory and execute the main script.
#### 1. K-Armed Bandits
This experiment compares the performance of five different bandit algorithms on the 10-armed testbed.
```bash
cd k_armed_bandits/
python visualise.py
```
This will generate Relative_algorithm_performance.png and Reward_Action_Graph.png in the results sub-folder.

#### 2. Dynamic Programming (Model-Based RL)

Gambler's Problem (Value Iteration): Solves for the optimal betting strategy given different probabilities of winning.
```bash
cd dynamic_programming/gamblers_problem/
python gambler.py
```

Grid World (Policy Iteration): Finds the optimal path through a maze-like grid.
```bash
cd 2_dynamic_programming/gridworld/
python deterministic_gridworld.py
```
#### 3. Model Free Methods
Blackjack (Monte Carlo): Uses Monte Carlo control to find the optimal policy for the game of Blackjack.
```bash
cd 3_model_free_methods/blackjack_mc/
python mc.py
```
Gym Environments (Q-Learning & SARSA): Solves for optimal policies in classic Gym environments like CliffWalking and FrozenLake.
```bash
cd 3_model_free_methods/gym_td_learning/
python q_learning.py  # Can be configured to run on different envs
python sarsa.py      # Can be configured to run on different envs
```

#### 4. Deep Q-Network (DQN) for Lunar Lander
This module trains a DQN agent on LunarLander-v3 and allows for hyperparameter experimentation. The script accepts three command-line arguments:

1. variable_to_test: (e.g., learning_rate, gamma, batch_size)
2. visualise: (True or False) - Renders the environment for a few episodes after training.
3. save_model_logs: (True or False) - Saves the final model weights and hyperparameters.

Example Usage: To run a hyperparameter sweep for learning_rate without visualization or saving logs
```bash
cd deep_q_learning/
python lunar_lander.py learning_rate False False
```