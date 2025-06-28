import matplotlib.pyplot as plt
import numpy as np

# default_world =  np.array([
#             ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
#             ['U', 'U', 'U', 'D', 'D', 'U', 'U', 'U', 'U', 'U', 'U', 'U'],
#             ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
#             ['S', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G'],
#         ])

# default_world = np.array([
#     ['R', 'R', 'R', 'R'],
#     ['S', 'C', 'C', 'G'],
# ])

default_world = np.array([
    ['R', 'L', 'R', 'D', 'L', 'R'],
    ['L', 'L', 'L', 'L', 'L', 'L'],
    ['S', 'C', 'C', 'C', 'C', 'G']

])
class Cliff():
    def __init__(self, world=default_world):
        self.world = world

        self.arrow_map = {
                            'U' : '⬆︎',
                            'D' : '⬇︎',
                            'L' : '⬅︎',
                            'R' : '➡︎',
                        }
        
        self.actions = {'U', 'D', 'L', 'R'}

        self.states = self.get_states()
        self.goal = tuple(np.argwhere(self.world == 'G').squeeze())
        self.start = tuple(np.argwhere(self.world == 'S').squeeze())

    def get_states(self):
        states = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i][j] != 'C' and self.world[i][j] != 'G':
                    states.append((i, j))

        return states
    
    def get_next_state_reward(self, current_state, action_char):
        next_r, next_c = current_state

        if action_char == 'U':
            next_r -= 1
        elif action_char == 'D':
            next_r += 1
        elif action_char == 'L':
            next_c -= 1
        elif action_char == 'R':
            next_c += 1

        if not (0 <= next_r < self.world.shape[0] and 0 <= next_c < self.world.shape[1]):   
            actual_next_s_tuple = tuple(current_state)
            reward = -10.0
        elif self.world[next_r, next_c] == 'C':
            actual_next_s_tuple = self.start
            reward = -100.0
        elif (next_r, next_c) == self.goal:
            actual_next_s_tuple = self.goal
            reward = 0.0
        else:
            actual_next_s_tuple = (next_r, next_c)
            reward = -1.0

        return actual_next_s_tuple, reward
        
        
    def show_policy_plot(self, current_policy_grid):
        cliff = current_policy_grid == 'C'

        bg = np.zeros_like(current_policy_grid, dtype=float)
        bg[cliff] = -10

        plt.imshow(bg, cmap='Blues_r', vmin=-10, vmax=10)
        plt.title(f"Cliff World Problem: Policy")

        for i in range(current_policy_grid.shape[0]):
            for j in range(current_policy_grid.shape[1]):
                cell_content = current_policy_grid[i, j]
                if cell_content == 'C':
                    continue
                elif cell_content == 'S' or cell_content == 'G':
                     plt.text(j, i, cell_content, ha='center', va='center', color='black', fontsize=60/current_policy_grid.shape[0] + 5)                    
                else:
                    arrow_char = self.arrow_map.get(cell_content, '?')
                    plt.text(j, i, arrow_char, ha='center', va='center', color='black', fontsize=60/current_policy_grid.shape[0] + 5)
        plt.show()

    def show_q_values_plot(self, current_policy_grid, Q_values):
        cliff = current_policy_grid == 'C'

        bg = np.zeros_like(Q_values, dtype=float)
        bg[cliff] = -10

        plt.imshow(bg, cmap='Blues_r', vmin=-10, vmax=10)
        plt.title(f"Cliff World Problem: Q values")

        for i in range(current_policy_grid.shape[0]):
            for j in range(current_policy_grid.shape[1]):
                cell_content = Q_values[i, j]
                if cell_content == 'C':
                    continue
                # elif cell_content == 'S' or cell_content == 'G':
                #      plt.text(j, i, cell_content, ha='center', va='center', color='black', fontsize=80/current_policy_grid.shape[0] + 5)                    
                else:
                    arrow_char = self.arrow_map.get(cell_content, '?')
                    plt.text(j, i, f'{cell_content:.2f}', ha='center', va='center', color='black', fontsize=20/current_policy_grid.shape[0] + 5)
        plt.show()

    def visualise_policy_q_values(self, current_policy_grid, Q_values):

        cliff = current_policy_grid == 'C'

        bg = np.zeros_like(current_policy_grid, dtype=float)
        bg[cliff] = -10

        plt.subplot(1, 2, 1)

        plt.imshow(bg, cmap='Blues_r', vmin=-10, vmax=10)
        plt.title(f"Cliff World Problem: Policy")

        for i in range(current_policy_grid.shape[0]):
            for j in range(current_policy_grid.shape[1]):
                cell_content = current_policy_grid[i, j]
                if cell_content == 'C':
                    continue
                elif cell_content == 'S' or cell_content == 'G':
                     plt.text(j, i, cell_content, ha='center', va='center', color='black', fontsize=60/current_policy_grid.shape[0] + 5)                    
                else:
                    arrow_char = self.arrow_map.get(cell_content, '?')
                    plt.text(j, i, arrow_char, ha='center', va='center', color='black', fontsize=60/current_policy_grid.shape[0] + 5)
        #plt.show()

        plt.subplot(1, 2, 2)
        plt.imshow(bg, cmap='Blues_r', vmin=-10, vmax=10)
        plt.title(f"Cliff World Problem: Q values")

        for i in range(current_policy_grid.shape[0]):
            for j in range(current_policy_grid.shape[1]):
                cell_content = Q_values[i, j]
                if cell_content == 'C':
                    continue
                # elif cell_content == 'S' or cell_content == 'G':
                #      plt.text(j, i, cell_content, ha='center', va='center', color='black', fontsize=80/current_policy_grid.shape[0] + 5)                    
                else:
                    arrow_char = self.arrow_map.get(cell_content, '?')
                    plt.text(j, i, f'{cell_content:.2f}', ha='center', va='center', color='black', fontsize=20/current_policy_grid.shape[0] + 5)
        plt.show()



