import matplotlib.pyplot as plt
import numpy as np
from policy_grid import policy_grid
policy_threshold = 0.01
gamma = 0.9

arrow_map = {
    'U' : '⬆︎',
    'D' : '⬇︎',
    'L' : '⬅︎',
    'R' : '➡︎',
}
actions = {'U', 'D', 'L', 'R'}
states_coords_np_list = np.argwhere(policy_grid=='U')
goal_coord_tuple = tuple(np.argwhere(policy_grid=='G').squeeze())

values = {tuple(s_coord): 0.0 for s_coord in states_coords_np_list}

def show_policy_plot(current_policy_grid):
    walls = current_policy_grid == 'W'
    goal_cell = current_policy_grid == 'G'

    bg = np.zeros_like(current_policy_grid, dtype=float)
    bg[walls] = -10
    bg[goal_cell] = 10

    plt.imshow(bg, cmap='Blues_r', vmin=-10, vmax=10)
    plt.title(f"Policy solving GridWorld problem\ngamma:{gamma}, slip probability:0")

    for i in range(current_policy_grid.shape[0]):
        for j in range(current_policy_grid.shape[1]):
            cell_content = current_policy_grid[i, j]
            if cell_content == 'W' or cell_content == 'G':
                continue
            else:
                arrow_char = arrow_map.get(cell_content, '?')
                plt.text(j, i, arrow_char, ha='center', va='center', color='black', fontsize=100/current_policy_grid.shape[0] + 5)
    plt.savefig("Policy solving GridWorld problem")
    plt.show()

def get_next_state_and_reward(current_s_np, action_char, grid, grid_goal_tuple):

    next_r, next_c = current_s_np

    if action_char == 'U':
        next_r -= 1
    elif action_char == 'D':
        next_r += 1
    elif action_char == 'L':
        next_c -= 1
    elif action_char == 'R':
        next_c += 1

    if not (0 <= next_r < grid.shape[0] and \
            0 <= next_c < grid.shape[1]) or \
            grid[next_r, next_c] == 'W':
        actual_next_s_tuple = tuple(current_s_np)
        immediate_r = -1.0
    elif (next_r, next_c) == grid_goal_tuple:
        actual_next_s_tuple = grid_goal_tuple
        immediate_r = 0.0
    else:
        actual_next_s_tuple = (next_r, next_c)
        immediate_r = -1.0

    return actual_next_s_tuple, immediate_r


def evaluate_policy():
    global values
    while True:
        delta = 0
        current_sweep_new_values = {}

        for state_np in states_coords_np_list:
            state_tuple = tuple(state_np)
            v_k_s = values[state_tuple]

            action_from_policy = policy_grid[*state_np]

            s_prime_tuple, immediate_r = get_next_state_and_reward(
                state_np, action_from_policy, policy_grid, goal_coord_tuple
            )

            v_k_s_prime = 0.0
            if s_prime_tuple != goal_coord_tuple:
                v_k_s_prime = values[s_prime_tuple]

            calculated_new_v_s = immediate_r + gamma * v_k_s_prime
            
            current_sweep_new_values[state_tuple] = calculated_new_v_s
            delta = max(delta, abs(v_k_s - current_sweep_new_values[state_tuple]))

        values = current_sweep_new_values
        
        if delta < policy_threshold:
            break

def improve_policy():
    global policy_grid
    policy_stable = True

    for state_np in states_coords_np_list:
        state_tuple = tuple(state_np)
        old_action = policy_grid[state_np[0], state_np[1]]
        
        action_q_values = {}

        for candidate_action in actions:
            s_prime_tuple, immediate_r = get_next_state_and_reward(
                state_np, candidate_action, policy_grid, goal_coord_tuple
            )

            v_pi_s_prime = 0.0
            if s_prime_tuple != goal_coord_tuple:
                v_pi_s_prime = values[s_prime_tuple]
            
            
            q_sa = immediate_r + gamma * v_pi_s_prime
            action_q_values[candidate_action] = q_sa

     
        best_action = max(action_q_values, key=action_q_values.get)
        
        policy_grid[*state_np] = best_action
        if old_action != best_action:
            policy_stable = False
            
    return policy_stable

i = 0
while True:
    print('Iteration ',i)
    evaluate_policy()
    if improve_policy():
        print(f"Policy stable after {i + 1} iterations.")
        break
    i += 1
    if i > 2000:
        print("Reached max iterations, stopping.")
        break

show_policy_plot(policy_grid)