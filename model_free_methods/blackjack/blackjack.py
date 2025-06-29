# Deck of cards: 13 cards each of 4 suits: clubs, diamonds, hearts, spades
# Each suit includes 3 face cards: King, Queen, Jack, and 10 number cards: 1 to 10. The 1 is known as an Ace
# The ace can also be used as an 11, known as usable ace, in Black Jack
'''
Rules:
      Try to get as close to 21 without going over.
      Kings, Queens, and Jacks are worth 10 points.
      Aces are worth 1 or 11 points.
      Cards 2 through 10 are worth their face value.
      (H)it to take another card.
      (S)tick to stop taking cards.
      In case of a tie, the bet is returned to the player.
      The dealer stops hitting at 17.
'''
logging = False
import numpy as np
class Dealer():
    def __init__(self):
        self.cards = set()
        #self.usable_ace = True
        self.balance = 0
        self.deck = {i for i in range(52)} # Cards are labelled 0 to 51, 13 cards for each suit
        self.current_sum = 0
        self.up_card = None
        #self.ace_count = 0
        #self.distribute_cards(player)

    def start_new_round(self, *args):
        self.cards.clear()
        self.deck = {i for i in range(52)}
        for player in args:
            player.cards.clear()
            player.episode_data.clear()
            #player.usable_ace = True
        self.distribute_cards(*args)


    def distribute_cards(self, *args):
        for person in args:
            random_cards = np.random.choice(a=list(self.deck), size=2, replace=False)
            person.cards.update(random_cards)
            self.deck.difference_update(random_cards)

        random_cards = np.random.choice(a=list(self.deck), size=2, replace=False)
        self.cards.update(random_cards)
        self.deck.difference_update(random_cards)

        up_card_raw = random_cards[0] % 13
        if up_card_raw  == 0:
            self.up_card = 1
        elif up_card_raw > 9:
            self.up_card = 10
        else:
            self.up_card = up_card_raw + 1

    def calculate_sum_of_cards(self):
        sum = 0
        ace = 0
        for card in self.cards:
            if card%13>9:
                sum+=10
            elif card%13>0:
                sum+=card%13+1
            else:
                ace+=1

        while(ace>0 and 21-(sum+ace)>=10):
            ace-=1
            sum+=11
        sum+=ace

        return sum
    

    def receive_bets(self, *args):
        for player in args:
            self.balance+=player.place_bet(10)

    def play_round(self, *args):

        for player in args:
            player.play_move(self.deck, self.up_card)
            if player.current_sum > 21:
                continue

        players_still_in = any(p.current_sum <= 21 for p in args)

        if players_still_in:
            while self.calculate_sum_of_cards() <= 16:
                new_card = np.random.choice(list(self.deck))
                self.deck.remove(new_card)
                if logging: print(f'Dealer hits. new card: {new_card}')
                self.cards.add(new_card)
        
        for player in args: 
            dealer_final_sum = self.calculate_sum_of_cards()

            if player.current_sum > 21:
                if logging: print('Player goes bust')
                player.assign_final_reward(-1)
            elif dealer_final_sum > 21:
                if logging: print('Dealer goes bust, returns money to non-busted players')
                player.receive_money(2*player.current_bet)
                self.balance-=2*player.current_bet
                player.assign_final_reward(+1)
            elif player.current_sum > dealer_final_sum:
                if logging: print(f'Player sum: {player.current_sum} > Dealer sum: {dealer_final_sum}. Player recieves 2x')
                player.receive_money(2*player.current_bet)
                self.balance-=2*player.current_bet
                player.assign_final_reward(+1)
            elif player.current_sum == dealer_final_sum:
                player.receive_money(player.current_bet)
                self.balance-=player.current_bet
                player.assign_final_reward(0)
                if logging: print(f"Push(Tie). Player gets their bet back.")
            else: 
                if logging: print(f"Dealer wins with {dealer_final_sum} against {player.current_sum}.")
                player.assign_final_reward(-1)
        

class Player():
    def __init__(self, initial_money, policy, epsilon=0):
        
        self.cards = set()
        self.balance = initial_money
        self.epsilon = epsilon
        #self.usable_ace = True
        #self.bust = False
        self.current_bet = 0
        self.policy = policy
        self.current_sum = 0
        self.episode_data = []
    
    def assign_final_reward(self, reward):
        for data in self.episode_data:
            data[2] = reward

    def calculate_sum_of_cards(self):
        sum = 0
        ace_value_1 = 0
        ace_value_11 = 0
        total_aces = 0
        for card in self.cards:
            if card%13>9:
                sum+=10
            elif card%13>0:
                sum+=card%13+1
            else:
                total_aces+=1
                ace_value_1+=1


        while(ace_value_1>0 and 21-(sum+ace_value_1)>=10):
            ace_value_1-=1
            sum+=11

        sum+=ace_value_1
        ace_value_11 = total_aces - ace_value_1

        return sum, ace_value_1, ace_value_11
    

    def hit(self, deck):
        new_card = np.random.choice(list(deck))
        deck.remove(new_card)
        self.cards.add(new_card)
        current_sum, _, _ = self.calculate_sum_of_cards()

        # if current_sum > 21 and self.usable_ace and self.ace_count > 0:
        #     current_sum-=10
        #     self.usable_ace = False

        self.current_sum = current_sum
        if logging: print(f'Players new card: {new_card} Player current sum:  {current_sum}')

    def place_bet(self, amount):
        self.current_bet = amount
        self.balance -=amount
        return amount
    
    def receive_money(self, amount):
        self.balance+=amount

    def play_move(self, deck, dealer_up_card):
        while True:
            self.current_sum, _ , ace_value_11 = self.calculate_sum_of_cards()
            if self.current_sum >= 21:
                break
            usable_ace = True if ace_value_11 >= 1 else False
            old_sum = self.current_sum
            n = np.random.random()
            if n < self.epsilon:
                move = np.random.choice(['h', 's'])
            else:
                move = self.policy[self.current_sum, usable_ace, dealer_up_card] 
            if logging: print('Player move: ', move)
            if move=='h':
                self.hit(deck=deck)
                self.episode_data.append([(old_sum, usable_ace, dealer_up_card), move, None])
                #if logging: print("Episode Data appended")
            elif move == 's':
                self.episode_data.append([(old_sum, usable_ace, dealer_up_card), move, None])
                break
        
            
