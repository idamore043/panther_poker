# Panther Poker

## Description
A Texas Hold'em simulation in Python, utilizing Q-Learning with Deep Q-Network (DQN) models to make optimal decisions. The simulation uses neural network-based decision-making for playing poker, utilizing Keras for the implementation of the models.
The simulation supports basic poker gameplay, including betting, checking, and folding as directed by player input. It places a particular emphasis on the mechanics of betting and card management.
The game is initialized with two players: Player 1 is controlled by user input via terminal, while Player 2 is managed by the AI. It is configured as a "five-ten" game, meaning the small blind is set at 5 chips and the big blind, contributed by the AI, is 10 chips.

## Prerequisites
To run this project, you will need the following:
```
- Python 3.6 or higher
- numpy
- Keras
```
Ensure you have the required packages installed. You can install them using pip:
```
pip install numpy keras
```

## Usage
To start the poker simulation, run the script from your command line from the correct directory:
```
python PokerAI.py
```

## Gameplay Instructions

### 1. Blind Betting and Card Dealing
If you choose to participate in the blind betting phase: Two private cards, known as **hole cards**, are dealt to each player. As Player 1, you will be prompted with the following options: <br />
> **(1) Bet:** You can bet any amount between 1 chip and the total number of chips you have available. <br />
> **(2) Check:** You can pass the action to the next player without making a bet, but only if no bet has been placed during the current round. <br />
> **(3) Fold:** You can fold your hand, which means forfeiting your chance to win the pot in the current game. Your opponent wins by default. <br />
### 2. Betting Restrictions
**Bet Limits:** Bets cannot be zero and cannot exceed your available stack of chips. <br />
**Active Bet Rule:** If there is an active bet, checking is not permitted; you must either call the bet, raise it, or fold. <br />
### 3. Game Progression
The game progresses through three consecutive and public card rounds following the initial betting: <br />
> **Flop:** Three community cards are dealt face-up on the table. <br />
> **Turn:** A fourth community card is added to the table. <br />
> **River:** A fifth and final community card is dealt. <br />

In each of these rounds, players have the opportunity to bet, check, or fold based on the developing strength of their hand in combination with the community cards.

### 4. Final Showdown and Determining the Winner
At the end of the River round, if more than one player remains, the game moves to a **showdown round** where all hands are revealed. <br />
The game evaluates the strength of each hand and calculates the win probabilities. <br />
The player with the strongest hand wins the pot. If there is a tie, the pot is evenly distributed among the winners.<br />
The player's stacks/chips are retained.<br />

## Classes
**'PokerAI'** <br />
Responsible for the AI behavior in the poker game. <br />
- Uses neural networks to learn and make decisions. <br />
- Implements Q-learning and uses a deep Q-network approach. <br />
- Uses experiences for training. <br />

**'PokerLogic'** <br />
Contains methods to evaluate poker hands and calculate win probabilities. <br />
- Hand Evaluation <br />
- Win Rate Estimation <br />
- Card Printing <br />

**'PokerGame'** <br />
Handles the mechanics of a poker game including: <br />
- Deck and Card Management <br />
- Simulated Betting Rounds <br />
- Player Actions <br />




