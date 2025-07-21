import numpy as np
from blackjack import Dealer, Player
import matplotlib.pyplot as plt

logging = False

dealer = Dealer()
if logging: print('Hello')
class MonteCarlo:
    def __init__(self, gamma = 1, initial_epsilon = 1.0, min_epsilon = 0.01, decay = 0.9999985):
        
        self.policy = {}
        self.gamma = gamma
        for dealer_card in range(1, 11):
            for s in range(2, 22):
                self.policy[(s, True, dealer_card)] = 'h'
            for s in range(2, 16):
                self.policy[(s, False, dealer_card)] = 'h'
            for s in range(16, 22):
                self.policy[(s, False, dealer_card)] = 's'

        self.player = Player(initial_money=100, policy=self.policy, epsilon = initial_epsilon)

        self.q_values = {
            ((sum, usable_ace, dealer_card), action) : np.random.random() for sum in range(2, 22) 
            for dealer_card in range(1, 11) for usable_ace in [True, False] for action in ['h', 's']
        }
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay   
        self.number_of_episodes = 0

        self.returns = {
            ((sum, usable_ace, dealer_card), action) : [] for sum in range(2, 22) 
            for dealer_card in range(1, 11) for usable_ace in [True, False] for action in ['h', 's']
        }

    def evaluate_and_improve_policy(self):
        episode_data = self.get_episode_data()
        G = 0
        visited_pairs = set()
        for data in episode_data[::-1]:
            G = self.gamma*G + data[2]
            S_t = data[0]
            A_t = data[1]
            if (S_t, A_t) not in visited_pairs:
                self.returns[(S_t, A_t)].append(G)
                self.q_values[(S_t, A_t)] = sum(self.returns[(S_t, A_t)])/len(self.returns[(S_t, A_t)])
                possible_q_values_actions = [(self.q_values[(S_t, action)], action) for action in ['h', 's']]
                self.policy[S_t] = max(possible_q_values_actions, key = lambda item: item[0])[1]
                visited_pairs.add((S_t, A_t))

        self.number_of_episodes += 1
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate


    def get_episode_data(self):
        player_end_balances = []
        self.player.policy = self.policy
        self.player.epsilon = self.epsilon 
        if logging: print('Current Policy : ', self.policy)
        if logging: print(f'Round aka episode starting....')
        dealer.start_new_round(self.player)
        self.player.balance = 100
        dealer.balance = 0
        if logging: print(f'Player 1 cards: {self.player.cards}, dealer cards: {dealer.cards}')
        dealer.receive_bets(self.player)
        if logging: print(f'Starting Round: Player 1 current balance: {self.player.balance}, Player 1 bet: {self.player.current_bet}, Dealer current balance: {dealer.balance}')
        dealer.play_round(self.player)
        if logging: print(f'End of Round: Player 1 current balance: {self.player.balance}, Dealer current balance: {dealer.balance}')
        player_end_balances.append(self.player.balance)
        if logging: print('Player 1 episode data: ', self.player.episode_data)
        
        return self.player.episode_data
    
    def visualise_policy(self):
        x = [i for i in range(1, 11)]
        pass
    
mc =  MonteCarlo()
for i in range(4000000):
    mc.evaluate_and_improve_policy()

final_policy = mc.policy
policy2 = {}
for d in range(1, 11):
    for s in range(2, 22):
        policy2[(s, True, d)] = 'h'
    for s in range(2, 17):
        policy2[(s, False, d)] = 'h'
    for s in range(17, 22):
        policy2[(s, False, d)] = 's'


player1_end_balances = []
player2_end_balances = []
print('Player 1 policy: ', final_policy)
print('Player 2 policy: ', policy2)

for j in range(20000):
    player1 = Player(initial_money=100, policy=final_policy)
    player2 = Player(initial_money=100, policy=policy2)
    dealer = Dealer()

    for i in range(10):
        print(f'Round {i+1} starting....')
        dealer.start_new_round(player1, player2)
        print(f'Player 1 cards: {player1.cards}, dealer cards: {dealer.cards}')
        print(f'Player 2 cards: {player2.cards}, dealer cards: {dealer.cards}')
        dealer.receive_bets(player1, player2)
        print(f'Starting Round: Player 1 current balance: {player1.balance}, Player 1 bet: {player1.current_bet}, Dealer current balance: {dealer.balance}')
        print(f'Starting Round: Player 2 current balance: {player2.balance}, Player 2 bet: {player2.current_bet}, Dealer current balance: {dealer.balance}')
        dealer.play_round(player1, player2)
        print(f'End of Round: Player 1 current balance: {player1.balance}, Dealer current balance: {dealer.balance}')
        print(f'End of Round: Player 2 current balance: {player2.balance}, Dealer current balance: {dealer.balance}')
    player1_end_balances.append(player1.balance)
    player2_end_balances.append(player2.balance)
    print("\n")

print("----")
print('Final Balance List for Player 1: ', player1_end_balances)
print('Average ending balance for Player 1: ', sum(player1_end_balances)/len(player1_end_balances))
print("----")
print('Final Balance List for Player 2: ', player2_end_balances)
print('Average ending balance for Player 2: ', sum(player2_end_balances)/len(player2_end_balances))
print("----")