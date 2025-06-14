import matplotlib.pyplot as plt
import numpy as np

policy_threshold = 0.01
gamma = 0.9

arrow_map = {
    'U' : '⬆︎',
    'D' : '⬇︎',
    'L' : '⬅︎',
    'R' : '➡︎',
}
actions = {'U', 'D', 'L', 'R'}


policy_grid = np.array([
    ['U','U','U','U','U','U','W','U','U','U'],
    ['U','W','U','W','W','U','W','U','W','W'],
    ['U','W','U','U','W','U','W','U','U','U'],
    ['U','U','W','U','W','U','U','W','W','U'],
    ['W','U','W','U','W','U','U','U','W','U'],
    ['U','U','W','U','U','U','U','U','U','U'],
    ['U','W','W','W','W','U','W','W','W','W'],
    ['U','U','U','U','W','U','W','U','U','U'],
    ['W','W','W','U','U','U','U','U','W','U'],
    ['U','U','U','U','U','U','U','U','W','G'],
    
])
states = np.argwhere(policy_grid=='U')
goal = np.argwhere(policy_grid=='G').squeeze()
values = {tuple(state): 0 for state in states}
new_values=values

def show_policy_plot(policy_grid):

    walls = policy_grid=='W'
    goal = policy_grid=='G'

    bg = np.zeros_like(policy_grid, dtype=float)
    bg[walls] = -10
    bg[goal] = 10

    plt.imshow(bg, cmap='Blues_r')
    plt.title("Policy solving GridWorld problem\ngamma:0.9, slip probability:0")

    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            if policy_grid[i][j]=='W' or policy_grid[i][j]=='G':
                continue
            else:
                plt.text(j, i, arrow_map[policy_grid[i,j]], ha='center', va='center', color='black', fontsize=12)
    plt.show()

def probability(s2, s1, policy_grid):
    p = 0
    if ((s2[0] == s1[0]-1) and (s2[1] == s1[1])) and (policy_grid[*s1]=='U'):
        p = 1
    elif ((s2[0] == s1[0]+1) and (s2[1] == s1[1])) and (policy_grid[*s1]=='D'):
        p = 1
    elif ((s2[0] == s1[0]) and (s2[1] == s1[1]-1)) and (policy_grid[*s1]=='L'):
        p = 1
    elif ((s2[0] == s1[0]) and (s2[1] == s1[1]+1)) and (policy_grid[*s1]=='R'):
        p = 1
    
    return p
def reward(s2):
    if tuple(s2) == tuple(goal):
        return 0
    return -1

def evaluate_policy():
    while True:
        delta = 0
        for state in states:
            v = values[tuple(state)]
            #values[state] = ∑p(s', r | s, 𝝅(s))[r + gamma*values[s']]
            tmp = 0
            for state2 in states:
                tmp+=probability(state2, state, policy_grid)*(reward(state2) + gamma*values[tuple(state2)])
            values[tuple(state)] = tmp
            delta = max(delta, abs(v - values[tuple(state)]))
        if delta < policy_threshold: break

def improve_policy():
    policy_stable = True
    for state in states:
        old_action = policy_grid[*state]
        exp = {'U': 0, 'D': 0, 'L': 0, 'R': 0,}
        #policy_grid[state] = argmaU_a ∑p(s', r | s, a)[r + gamma*values[s']]
        for action in actions:
            for state2 in states:
                policy_grid_copy = policy_grid.copy()
                policy_grid_copy[*state] = action
                exp[action] += probability(state2, state, policy_grid_copy)*(reward(state2) + gamma*values[tuple(state2)])

            policy_grid[*state] = max(exp, key=exp.get)
        if old_action != policy_grid[*state]: policy_stable=False

    return policy_stable
i = 0
while True:
    print('Iteration ', i)
    evaluate_policy()
    if improve_policy(): break
    i+=1

show_policy_plot(policy_grid)