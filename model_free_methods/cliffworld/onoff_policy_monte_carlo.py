import numpy as np
from cliff import Cliff

# Class for on policy first visit MC prediction and control, achieved via an epsilon-greedy policy
class MonteCarlo():
    
    def __init__(self, world, epsilon = 0.1, gamma = 0.9):
        self.epsilon = epsilon
        self.world = world
        self.gamma = gamma

        self.q_values = {
            (state, action): np.random.random() for action in self.world.actions for state in self.world.states
        }

        self.returns = {
            (state, action): [] for action in self.world.actions for state in self.world.states
        }

        self.policy = self.get_policy()

        self.state_count = {
            state: 0 for state in self.world.states
        }

    def get_policy(self):
        policy = {state: None for state in self.world.states}
        for state in self.world.states:
            n = np.random.random()
            # if n < self.epsilon:
            #     policy[state] = np.random.choice(list(self.world.actions))
            # else:
            possible_q_values = [(action, self.q_values[(state, action)]) for action in self.world.actions]
            policy[state] = max(possible_q_values, key=lambda item: item[1])[0]
                
        return policy
    
    def improve_policy(self):
        episode, t = self.generate_episode()
        G = 0
        t-=1
        visited_pairs = set()
        while t >= 0:
            G = self.gamma*G + episode[t][2]
            S_t, A_t = episode[t][0], episode[t][1]

            #previous_state_action_pairs = [(episode[i][0], episode[i][1]) for i in range(t-1)]

            if (S_t, A_t) not in visited_pairs:
                self.returns[(S_t, A_t)].append(G)
                self.q_values[(S_t, A_t)] = sum(self.returns[(S_t, A_t)])/len(self.returns[(S_t, A_t)])
                visited_pairs.add((S_t, A_t))
            t-=1
            
        return self.get_policy()

    def generate_episode(self):
        episode = []
        #print('59: ', self.world.states)
        S_current = self.world.states[np.random.randint(0, len(self.world.states))]
        #print('61: ', S_current)
        #A_current = self.policy[S_current]
        R_next = +1
        
        for _ in range(200):
            n = np.random.random()
            if n < self.epsilon:
                A_current = np.random.choice(list(self.world.actions))
            else: 
                A_current = self.policy[S_current]
                # q_vals_for_state = {action: self.q_values[(S_current, action)] for action in self.world.actions}
                # A_current = max(q_vals_for_state, key=q_vals_for_state.get)
            S_next, R_next = self.world.get_next_state_reward(current_state = S_current, action_char = A_current)
            self.state_count[S_current]+=1
            #A_next = self.policy[S_next]
            episode.append((S_current, A_current, R_next))

            if S_next == self.world.goal or R_next == -10:
                break
            # print(f'(S_t, A_t, R_t+1): ({S_current}, {A_current}, {R_next})')
            
            S_current = S_next
            #A_current = A_next

            
        #print(f'Episode with {len(episode)} timesteps generated')
        return episode, len(episode)
    
    def find_optimal_policy(self, iterations = 10000):
        i = 0
        
        while i < iterations:
            new_policy = self.improve_policy()
            if i%1000==0: print(f'Iteration {i}')
            i+=1
            self.policy = new_policy
            #self.epsilon = self.epsilon*(1 - 0.6*i/iterations) 

        print('Policy Converged')

    def convert_policy_to_map(self):
        world_map = np.array(self.world.world)

        for state in self.policy.keys():
            world_map[state] = self.policy[state]

        return world_map
    
    def convert_q_values_to_map(self):
        q_map = np.zeros_like(self.world.world, dtype=float)

        for state in self.policy.keys():
            #
            # for action in self.world.actions:
            #print(self.q_values.items())
            possible_q_values = [self.q_values[(state, action)] for action in self.world.actions]
            #print(f'Possible q valued for state {state}: {possible_q_values}')
            q_map[state] = max(possible_q_values)
            #print(q_map[state])
        return q_map


        
cliff = Cliff()     
cliff.show_policy_plot(cliff.world)   
mc = MonteCarlo(cliff)

mc.find_optimal_policy(iterations=100000)
print('Policy: ', mc.policy)
print('State count: ', mc.state_count)
world_map = mc.convert_policy_to_map()
cliff.show_policy_plot(world_map)

q_map = mc.convert_q_values_to_map()
#print(f'Q values: {mc.q_values}')
#print()
#print(f'Q map: ', q_map)
cliff.show_q_values_plot(current_policy_grid=world_map, Q_values=q_map)

#cliff.visualise_policy_q_values(current_policy_grid=world_map, Q_values=q_map)