import json
import atexit
import signal
import random
import sys
import numpy as np
from tables import flushes
from tables import unique5
from tables import hash_values
from tables import hash_adjust
from collections import defaultdict
from itertools import combinations

class PokerAI:

    def __init__(self, action_size):
        self.nodeMap = {}
        self.nash_equilibrium = dict()
        # Assuming four actions: call, raise 25, raise 50, fold.
        self.action_size = action_size
        self.regret_sum = defaultdict(lambda: np.zeros(self.action_size))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.action_size))
        self.initial_strategy = np.array([0.4, 0.3, 0.2, 0.1])
        self.action_names = ['call', 'raise_25', 'raise_50', 'fold']
        self.strategy = defaultdict(lambda: np.ones(self.action_size) / self.action_size)

        atexit.register(self.save_strategy, 'ai_strategy_at_exit.json')
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def train(self, iterations, save_interval=1000, save_path='ai_strategy.json'):
        for i in range(iterations):
            print(f"Starting training iteration: {i + 1}")
            game = PokerGame(players=2, poker_ai=self)
            game_history = game.play()  # Play a full game.
            for state, action, payoff in game_history:
                self.update_strategy(state, action, payoff)
            if (i + 1) % save_interval == 0:
                print(f"Saving AI strategy at iteration {i + 1}")
                self.save_strategy(save_path)
            print("Current iteration completed. Checking game over status...")
            if game.game_over():
                print("Game ended, resetting for new game.")
            else:
                print("Unexpected game continuation.")

            print(f"Ending training iteration: {i + 1}\n")

    def save_strategy(self, filepath):
        data = {
            'regret_sum': {str(k): v.tolist() for k, v in self.regret_sum.items()},
            'strategy_sum': {str(k): v.tolist() for k, v in self.strategy_sum.items()}
        }
        with open(filepath, 'w') as f:
            for key, value in data.items():
                json.dump({key: value}, f)
                f.write('\n')  # Ensure each entry is on a new line.
                print(f"Saved {key}: {value}")  # Print each saved line.

    def load_strategy(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.regret_sum = defaultdict(lambda: np.zeros(self.action_size),
                                      {k: np.array(v) for k, v in data['regret_sum'].items()})
        self.strategy_sum = defaultdict(lambda: np.zeros(self.action_size),
                                        {k: np.array(v) for k, v in data['strategy_sum'].items()})

    def handle_exit(self, signum, frame):
        print(f"Exiting due to signal {signum}, saving strategy.")
        self.save_strategy('ai_strategy_on_signal.json')
        sys.exit(0)

    def get_strategy(self, state):
        if state not in self.regret_sum:
            # Initialize with initial strategy if not present.
            self.regret_sum[state] = np.zeros(self.action_size)
            self.strategy[state] = self.initial_strategy.copy()

        positive_regrets = np.maximum(self.regret_sum[state], 0)
        norm_sum = positive_regrets.sum()
        if norm_sum > 0:
            self.strategy[state] = positive_regrets / norm_sum
        else:
            self.strategy[state] = np.ones(self.action_size) / self.action_size  # Fallback to uniform distribution.

        self.strategy_sum[state] += self.strategy[state]
        return self.strategy[state]

    def get_action(self, state):
        strategy = self.get_strategy(state)
        action_index = np.random.choice(self.action_size, p=strategy)
        # Define a mapping from action indices to action names.
        action_names = {0: 'call', 1: 'raise_25', 2: 'raise_50', 3: 'fold'}
        # Use the action index to get the corresponding action name.
        print(f"Chosen action: {action_names[action_index]}")
        return action_names[action_index]

    def update_strategy(self, state, action_taken, payoff):
        # If the state is new, initialize strategy and regret arrays.
        if state not in self.regret_sum:
            self.regret_sum[state] = np.zeros(self.action_size)
            self.strategy[state] = self.initial_strategy.copy()  # Use initial strategy as a starting point.

        strategy = self.get_strategy(state)
        action_index = self.action_names.index(action_taken)

        # Calculate the expected payoff for the strategy.
        payoff_array = np.full(self.action_size, payoff)
        expected_payoff = np.dot(strategy, payoff_array)

        # Calculate regret for each action.
        regret = payoff_array - expected_payoff

        # Update regrets and strategy sums.
        self.regret_sum[state] += strategy * regret  # Update regret based on the probability of taking each action.
        self.strategy_sum[state] += strategy

        print(f"Updated regrets for state {state}: {self.regret_sum[state]}")
        print(f"Strategy for state {state}: {self.strategy[state]}")

class PokerLogic:

    # Constants for encoding
    def __init__(self):
        self.rank_map = {
            '2': (2, 0), '3': (3, 1), '4': (5, 2), '5': (7, 3),
            '6': (11, 4), '7': (13, 5), '8': (17, 6), '9': (19, 7),
            '10': (23, 8), 'J': (29, 9), 'Q': (31, 10), 'K': (37, 11), 'A': (41, 12)
        }
        self.suit_map = {
            'Clubs': 0x8000, 'Diamonds': 0x4000, 'Hearts': 0x2000, 'Spades': 0x1000
        }

    def convert_and_encode_hand(self, hand):
        encoded_cards = []
        for card in hand:
            prime, rank_index = self.rank_map[card['value']]
            suit = self.suit_map[card['suit']]
            rank_binary = 1 << (rank_index + 16)  # Starting from bit 16.
            binary_representation = rank_binary | suit | (rank_index << 8) | prime
            encoded_cards.append(binary_representation)
            # print(f"  Combined binary: {bin(binary_representation)}, decimal: {binary_representation}").
        return encoded_cards

    def calculate_hand_value(self, encoded_hand):
        q = 0
        for card in encoded_hand:
            q |= card
        q >>= 16  # Shift to right to isolate the rank bits.

        SUIT_MASK = 0xF000  # Mask to isolate the suit bits.
        # Check if all cards have the same suit.
        if all((card & SUIT_MASK) == (encoded_hand[0] & SUIT_MASK) for card in encoded_hand):
            return flushes[q]  # Return the strength of flushes or straight flushes.

        # Check for Straights and High Card hands.
        if unique5[q]:
            return unique5[q]

        # If not found in unique5, then calculate using hash tables.
        product = 1
        for card in encoded_hand:
            product *= (card & 0xff)  # Product of the rank bits of the card.

        hand_rank_index = self.find_fast(product)

        return hash_values[hand_rank_index]

    def find_fast(self, u):
        original_u = u  # Store original u for debugging.
        u = (u + 0xe91aaa35) & 0xffffffff  # Emulate 32-bit unsigned addition.
        u ^= u >> 16
        u = (u + (u << 8)) & 0xffffffff  # Emulate 32-bit unsigned addition.
        u ^= u >> 4
        b = ((u >> 8) & 0x1ff)
        a = ((u + (u << 2)) & 0xffffffff) >> 19  # Emulate 32-bit unsigned addition then shift.
        r = a ^ hash_adjust[b]
        # print(f"find_fast - u: {original_u}, a: {a}, b: {b}, r: {r}")
        return r

    def eval_7hand(self, hand):
        best_value = 9999  # Use a high value that will be overwritten by any valid poker hand value.
        best_hand = None
        for combo in combinations(hand, 5):  # Generates all 5-card combinations from the 7-card hand.
            converted_hand = self.convert_and_encode_hand(combo)
            hand_value = self.calculate_hand_value(converted_hand)
            if hand_value < best_value:
                best_value = hand_value
                best_hand = combo
        return best_value, best_hand

    def evaluate_hand(self,
                      hole_cards):  # Evaluate the strength of a poker hand given the hole cards and community cards.
        # Ensure hole_cards is a list.
        if isinstance(hole_cards, dict):
            hole_cards = [hole_cards]

        # Combine hole cards with community cards to form the full hand.
        full_hand = hole_cards  # + community_cards

        # Separate cards by suit and value.
        cards_by_suit = {}
        cards_by_value = {}

        # Evaluate the strength of the hand.
        for card in full_hand:
            suit = card['suit']
            value = card['value']

            if suit not in cards_by_suit:
                cards_by_suit[suit] = []
            cards_by_suit[suit].append(value)

            if value not in cards_by_value:
                cards_by_value[value] = []
            cards_by_value[value].append(suit)

        # Check for Royal Flush
        for suit, values in cards_by_suit.items():
            if '10' in values and 'J' in values and 'Q' in values and 'K' in values and 'A' in values:
                return 'Royal Flush', 10

        # Check for Straight Flush
        for suit, values in cards_by_suit.items():
            straight_values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            for i in range(len(straight_values) - 4):
                if all(value in values for value in straight_values[i:i + 5]):
                    return 'Straight Flush', 9

        # Check for Four of a Kind
        for value, suits in cards_by_value.items():
            if len(suits) >= 4:
                return 'Four of a Kind', 8

        # Check for Full House
        has_three = False
        has_pair = False
        for value, suits in cards_by_value.items():
            if len(suits) >= 3:
                has_three = True
            elif len(suits) >= 2:
                has_pair = True
        if has_three and has_pair:
            return 'Full House', 7

        # Check for Flush
        for suit, values in cards_by_suit.items():
            if len(values) >= 5:
                return 'Flush', 6

        # Check for Straight
        for value, suits in cards_by_value.items():
            straight_values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            for i in range(len(straight_values) - 4):
                if all(suit in cards_by_suit and value in cards_by_suit[suit] for value in straight_values[i:i + 5]):
                    return 'Straight', 5

        # Check for Three of a Kind
        for value, suits in cards_by_value.items():
            if len(suits) >= 3:
                return 'Three of a Kind', 4

        # Check for Two Pair
        num_pairs = 0
        for value, suits in cards_by_value.items():
            if len(suits) >= 2:
                num_pairs += 1
        if num_pairs >= 2:
            return 'Two Pair', 3

        # Check for One Pair
        for value, suits in cards_by_value.items():
            if len(suits) >= 2:
                return 'One Pair', 2

        # If none of the above, it's High Card
        return 'High Card', 1

    def simple_card_print(self, cards):  # Simple card printing method for debugging and testing.
        suits_ascii = {'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣', 'Spades': '♠'}
        values_ascii = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
                        '7': '7', '8': '8', '9': '9', '10': '10',
                        'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}

        card_strings = []
        for card in cards:
            suit = card['suit']
            value = card['value']
            card_str = f"{values_ascii[value]}{suits_ascii[suit]}"
            card_strings.append(card_str)

        print(" ".join(card_strings))

    def estimate_win_rate(self, hand_strength):  # Estimate the win rate based on the strength of the hand.
        # Rough probabilities of winning based on hand strength.
        win_rate = {
            10: 0.90,  # Royal Flush
            9: 0.80,  # Straight Flush
            8: 0.70,  # Four of a Kind
            7: 0.60,  # Full House
            6: 0.50,  # Flush
            5: 0.40,  # Straight
            4: 0.30,  # Three of a Kind
            3: 0.20,  # Two Pair
            2: 0.10,  # One Pair
            1: 0.05  # High Card
        }
        return win_rate.get(hand_strength, 0)

    def hand_rank(self, hand):  # Return a numerical rank for a given hand (e.g., 'Two Pair' -> 3).
        rank_order = {
            'High Card': 0,
            'One Pair': 1,
            'Two Pair': 2,
            'Three of a Kind': 3,
            'Straight': 4,
            'Flush': 5,
            'Full House': 6,
            'Four of a Kind': 7,
            'Straight Flush': 8,
            'Royal Flush': 9
        }
        return rank_order[hand]


class PokerGame:
    def __init__(self, players, poker_ai):  # Initialize a poker game with the given number of players.
        self.continue_game = True  # New attribute to keep playing when player fold.
        self.player_out = False  # New flag to check if player is out of chips.
        self.poker_ai = poker_ai
        self.deck = self.create_deck()
        self.shuffle_deck()
        self.players = players
        self.community_cards = []
        # Modified the player to have 2 additional parameters hand strength that will vary from 10 to 1 and a very basic win rate change.
        self.hands = {player: {'cards': [], 'current_bet': 0} for player in
                      range(players)}

        self.pot = 0
        self.player_chips = [1000] * players  # Initialize player chips with 1000 for each player.
        self.small_blind = 5  # Small blind amount
        self.big_blind = 2 * self.small_blind  # Big blind amount.
        self.poker_logic = PokerLogic()
        self.current_bet = 0  # Initialize current bet to zero.
        self.player_label = "Player 1"  # Define Player's Name.
        self.ai_label = "Player 2"  # Define AI's Name.
        self.actions_this_round = []
        self.game_history = []
        self.action_sequence = ""
        self.terminal_sequences = {"cc", "crc", "rRc", "rrc"}
        self.reset_for_new_round()


    def record_action(self, action):
        action_codes = {'check': 'c', 'raise_25': 'r', 'raise_50': 'R', 'fold': 'f'}
        self.action_sequence += action_codes.get(action, "")

    def is_terminal_sequence(self):
        return self.action_sequence in self.terminal_sequences

    def create_deck(self):  # Create a standard deck of 52 playing cards.
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        #values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        values = ['8', '9', '10', 'J', 'Q', 'K', 'A']
        return [{'suit': suit, 'value': value} for suit in suits for value in values]

    ##################################################################################################
    
    def get_state(self, player_index):
        player_cards = tuple(card['value'] for card in self.hands[player_index]['cards'])
        community_cards = tuple(card['value'] for card in self.community_cards)
        current_bets = tuple(hand['current_bet'] for hand in self.hands.values())
        pot_size = self.pot
        return (player_cards, community_cards, pot_size, current_bets)

    def simulate_private_card_betting_round(self):
        print("\n\033[1mPrivate Card Betting Round:\033[0m")
        player_cards = self.hands[0]['cards']
        action_index, raise_amount = self.poker_ai.get_private_round_action(player_cards)

        if action_index == 1 or action_index == 2:  # Raise action taken by AI.
            print(f"AI raises by {raise_amount} chips.")
            self.pot += raise_amount
            self.player_chips[1] -= raise_amount  # Assuming AI has sufficient chips.

            player_decision = input(f"Do you want to 'call' {raise_amount} chips or 'fold'? ")
            if player_decision == 'call':
                self.player_chips[0] -= raise_amount  # Assuming player has sufficient chips.
                self.pot += raise_amount
                print(f"{self.player_label} calls.")
                result = raise_amount  # Update result for call.
            elif player_decision == 'fold':
                print(f"{self.player_label} folds. {self.ai_label} wins the pot.")
                self.continue_game = False
                result = 0  # Update result for fold.

            self.update_game_history(player_cards, raise_amount, result)

    #################################################################################################

    def shuffle_deck(self):
        random.shuffle(self.deck)
        print(f"Deck shuffled. Deck size: {len(self.deck)}")  # Debugging output

    def update_hs_and_wr(self):
        for player in range(self.players):
            # Evaluate the current hand strength.
            hand_strength_info = self.poker_logic.evaluate_hand(self.hands[player]['cards'])
            self.hands[player]['hand_strength'] = hand_strength_info[1]
            # Update win rates based on the new hand strength.
            self.hands[player]['win_rate'] = self.poker_logic.estimate_win_rate(hand_strength_info[1])

    def deal_hole_cards(self):
        if len(self.deck) < self.players * 2:
            raise ValueError("Not enough cards to deal hole cards to all players.")
        print(f"Dealing hole cards... Deck size before: {len(self.deck)}")
        for _ in range(2):
            for player in range(self.players):
                card = self.deck.pop()
                self.hands[player]['cards'].append(card)
                print(f"Player {player + 1} receives card: {card['value']} of {card['suit']}")
        print("Hole cards dealt to all players.")

    def deal_community_cards(self):
        if len(self.deck) < 3:
            raise ValueError("Not enough cards to deal community cards.")
        for _ in range(3):
            card = self.deck.pop()
            self.community_cards.append(card)
            # Print each community card as it is dealt.
            print(f"Community card dealt: {card['value']} of {card['suit']}")
        print("All community cards dealt.")

    def print_stacks(self):  # Print the current chip stacks for each player.
        print("\033[1mPlayer Stacks:\033[0m")
        for player, chips in enumerate(self.player_chips):
            print(f"Player {player + 1} has {chips} chips.")

    def print_pot(self):  # Print the current total pot.
        print("\033[1mCurrent Pot:\033[0m")
        print(f"{self.pot} chips.")

    def distribute_pot(self, winners):  # Distribute the pot among the winning players.
        share = self.pot // len(winners)
        for winner in winners:
            self.player_chips[winner] += share
        self.pot = 0

    def showdown(self):
        try:
            # Evaluate the best hand for each player.
            for player in range(self.players):
                best_hand_value, best_hand_cards = self.poker_logic.eval_7hand(
                    self.hands[player]['cards'] + self.community_cards)
                self.hands[player]['best_hand_value'] = best_hand_value
                self.hands[player]['best_hand_cards'] = best_hand_cards

            # Determine and print the winner
            best_hand_value = min(hand['best_hand_value'] for hand in self.hands.values())
            winners = [player for player, hand in self.hands.items() if hand['best_hand_value'] == best_hand_value]
            self.distribute_pot(winners)

            for player in range(self.players):
                print(f"Player {player + 1}'s best hand value: {self.hands[player]['best_hand_value']}")
                print(
                    f"Player {player + 1}'s best hand cards: {' '.join(card['value'] + card['suit'][0] for card in self.hands[player]['best_hand_cards'])}")
            print(f"Winner(s): {', '.join('Player ' + str(winner + 1) for winner in winners)}")
        except Exception as e:
            print(f"An error occurred in showdown: {str(e)}")
        finally:
            print("Showdown completed.")

        return self.game_history


    def simulate_blind_round(self):
        print("\n\033[1mBlind Betting:\033[0m")
        while True:
            action = input(f"{self.player_label}, do you want to post the small blind (5 chips)? (yes/no): ")
            if action == "yes":
                small_blind_amount = min(self.small_blind, self.player_chips[0])
                self.player_chips[0] -= small_blind_amount
                self.pot += small_blind_amount
                print(f"{self.player_label} posts small blind of {small_blind_amount} chips.")
                break
            elif action == "no":
                print(f"{self.player_label} opts not to post the small blind.")
                # AI automatically wins if player doesn't post small blind.
                print(f"{self.ai_label} wins by default!")
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # AI posts big blind.
        big_blind_amount = min(self.big_blind, self.player_chips[1])
        # Subtract big blind from AI chips.
        self.player_chips[1] -= big_blind_amount
        # Add big blind to pot.
        self.pot += big_blind_amount
        print(f"{self.ai_label} posts big blind of {big_blind_amount} chips.")
        self.print_stacks()
        self.print_pot()

    ###############################################################################################################
    
    def simulate_blind_round_ai(self):
        print("\n\033[1mBlind Betting:\033[0m")
        # Automatically post the small blind from Player 1's chips.
        small_blind_amount = min(self.small_blind, self.player_chips[0])
        self.player_chips[0] -= small_blind_amount
        self.pot += small_blind_amount
        print(f"{self.player_label} posts small blind of {small_blind_amount} chips.")

        # Automatically post the big blind from AI's chips.
        big_blind_amount = min(self.big_blind, self.player_chips[1])
        self.player_chips[1] -= big_blind_amount
        self.pot += big_blind_amount
        print(f"{self.ai_label} posts big blind of {big_blind_amount} chips.")

        # Print the current status of chip stacks and the pot.
        self.print_stacks()
        self.print_pot()

    ###############################################################################################################
    
    def play_private_round(self):  # Method for private card rounds of poker.
        print("\n\033[1mDealt Private Cards to Players:\033[0m")
        self.deal_hole_cards()
        self.simulate_betting_round_ai()
        self.print_stacks()
        self.print_pot()

    def play_community_round(self, round_name, community_card_count):  # Method for community card rounds of poker.
        print(f"\n\033[1m{round_name}:\033[0m")
        self.deal_community_card(community_card_count)
        self.simulate_betting_round_ai()
        self.print_stacks()
        self.print_pot()

    def reset_for_new_round(self):
        self.deck = self.create_deck()  # This should recreate a full deck.
        self.shuffle_deck()  # Reshuffle the newly created deck.
        self.community_cards = []
        self.pot = 0  # Reset the pot.
        self.player_out = False  # Reset player out condition if used.
        self.hands = {player: {'cards': [], 'current_bet': 0, 'folded': False} for player in range(self.players)}
        print("Deck reset and shuffled. Deck size: {}".format(len(self.deck)))  # Debugging statement.

    def play(self):
        self.game_history = []
        self.reset_for_new_round()  # Ensure starting from a fresh state.

        while not self.game_over():
            print("Starting new hand...")
            self.deal_hole_cards()
            self.simulate_betting_round_ai()  # Simulate betting after dealing hole cards.

            if not self.continue_game:
                print("Game stopped after private round.")
                break

            self.deal_community_cards()  # Dealing the flop.
            self.simulate_betting_round_ai()

            if not self.continue_game:
                print("Game stopped after flop.")
                break

            # If additional rounds are desired:
            # self.deal_turn(); self.deal_river() etc.

            self.showdown()  # Run showdown to determine the winner.
            print("Showdown completed.")
            self.reset_for_new_round()

        print("Game over condition met.")
        return self.game_history

    #####################################################################################################################################

    def simulate_betting_round_ai(self):
        player_actions_completed = [False] * self.players

        while not all(player_actions_completed):
            for player in range(self.players):
                if not player_actions_completed[player]:
                    state = self.get_state(player)  # Get state for the current player.

                    if player == 0:
                        # AI Player (Player 1).
                        action = self.poker_ai.get_action(state)
                    else:
                        # Dummy Bot (Player 2).
                        action = self.get_player2_action(self.current_bet)

                    self.record_action(action)
                    result = self.execute_action(player, action)
                    player_actions_completed[player] = True
                    if player == 0:
                        self.game_history.append((state, action, result))

                    if self.is_terminal_sequence():
                        print(f"Terminal sequence reached with {self.action_sequence}. Ending betting round.")
                        return

        print("Betting round completed.")

    def execute_action(self, player, action):
        if action == 'call':
            amount_to_call = self.current_bet - self.hands[player]['current_bet']
            self.player_chips[player] -= amount_to_call
            self.hands[player]['current_bet'] += amount_to_call
            self.pot += amount_to_call
            return amount_to_call

        elif action == 'raise_25':
            raise_amount = 25
            self.player_chips[player] -= raise_amount
            self.hands[player]['current_bet'] += raise_amount
            self.current_bet = max(self.current_bet, self.hands[player]['current_bet'])
            self.pot += raise_amount
            return raise_amount

        elif action == 'raise_50':
            raise_amount = 50
            self.player_chips[player] -= raise_amount
            self.hands[player]['current_bet'] += raise_amount
            self.current_bet = max(self.current_bet, self.hands[player]['current_bet'])
            self.pot += raise_amount
            return raise_amount

        elif action == 'fold':
            self.hands[player]['folded'] = True
            return 0

    # Dummy bot for player 2
    def get_player2_action(self, current_bet):
        # Choose to raise 25, call, or check based on the probability.
        rand_choice = random.random()
        if rand_choice < 0.25:
            return 'check'
        elif rand_choice < 0.75:
            return 'call'
        else:
            if current_bet == 0:
                return 'raise_25'
            else:
                return 'call'

    #####################################################################################################################################
    
    def get_player_action(self, player, current_bet):
        print(f"\033[1m{self.player_label}'s Turn:\033[0m")
        while True:
            if current_bet == 0:
                print(f"{self.player_label}, would you like to 'call', 'raise', or 'fold'.")
                response = input("Select your action: ")
            else:
                print(f"{self.player_label}, would you like to 'call', 'raise', or 'fold'.")
                response = input("Select your action: ")
            if response in ["call", "raise", "fold", "check"]:
                return response
            print("Invalid input.")

    def get_raise_amount(self, player, current_bet):
        while True:
            print(f"{self.player_label}, choose your bet amount:")
            print("1: Call the current bet.")
            print("2: Raise by 25 chips.")
            print("3: Raise by 50 chips.")
            choice = input("Select option '1', '2', or '3': ")
            if choice == '1':
                return current_bet  # Player chooses to call.
            elif choice == '2':
                if self.player_chips[player] >= current_bet + 25:
                    return current_bet + 25  # Player chooses to raise by 25.
                else:
                    print("Not enough chips to raise by 25.")
            elif choice == '3':
                if self.player_chips[player] >= current_bet + 50:
                    return current_bet + 50  # Player chooses to raise by 50.
                else:
                    print("Not enough chips to raise by 50.")
            else:
                print("Invalid input.")

    def player_bet(self, player, bet_amount):
        if bet_amount < 0:
            print("Error: Bet amount cannot be negative.")
            return
        if bet_amount > self.player_chips[player]:
            print("Not enough chips. You cannot bet more than you have.")
            return
        self.player_chips[player] -= bet_amount
        self.hands[player]['current_bet'] += bet_amount
        self.pot += bet_amount
        print(f"{self.player_label} bets {bet_amount} chips.")
        self.current_bet = max(self.current_bet, self.hands[player]['current_bet'])
        self.actions_this_round.append(('bet', player, bet_amount))

    def player_checks(self, bet_active):
        if bet_active:
            raise ValueError(f"{self.player_label} cannot check when a bet has been made.")
        else:
            print(f"{self.player_label} checks.")

    def player_fold(self, player):
        # Player folds
        print(f"{self.player_label} folds.")
        self.hands[player]['folded'] = True
        other_player = next(player for player in range(self.players) if not self.hands[player].get('folded', False))
        print(f"Player {other_player + 1} wins by default!")
        self.distribute_pot([other_player])
        self.continue_game = False
        self.reset_for_new_round()

        self.actions_this_round.append(('bet', player))

    def game_over(self):
        if any(chips <= 0 for chips in self.player_chips):
            print(f"Game over! At least one player is out of chips.")
            return True
        return False

    def ai_bet(self, player, bet_amount):
        self.ai_label = f"Player {player + 1}"
        if bet_amount > self.player_chips[player]:
            print(
                f"{self.ai_label} does not have enough chips to bet {bet_amount}. Current chips: {self.player_chips[player]}")
            bet_amount = self.player_chips[player]
        self.player_chips[player] -= bet_amount
        self.hands[player]['current_bet'] += bet_amount
        self.pot += bet_amount
        print(f"{self.ai_label} bets {bet_amount} chips.")

    def ai_check(self, player):
        if self.hands[player].get('current_bet', 0) == 0:
            print(f"{self.ai_label} checks. No bet to match.")
        else:
            print(f"{self.ai_label} cannot check because there is a bet to match.")

    def ai_fold(self, player):
        # AI folds.
        print(f"{self.ai_label} folds.")

        # Set the fold status for the player.
        self.hands[player]['folded'] = True

        # Find the remaining player who hasn't folded.
        other_player = next(player for player in range(self.players) if not self.hands[player].get('folded', False))
        print(f"Player {other_player + 1} wins by default!")
        self.distribute_pot([other_player])

        # End the game.
        self.continue_game = False
        self.reset_for_new_round()

    def update_game_history(self, state, action, result):
        self.game_history.append((state, action, result))


if __name__ == "__main__":

    action_size = 4  # For example, 0: call, 1: raise 25, 2: raise 50, 3: fold.
    poker_ai = PokerAI(action_size=action_size)
    poker_ai.train(10000)