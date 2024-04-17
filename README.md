# Panther Poker

## Description
A Texas Hold'em simulation in Python, utilizing CFR models to make optimal decisions.
The simulation supports basic poker gameplay, including raising, checking, and folding as directed by player input. The game is initialized with two players: Player 1 is controlled by user input via terminal, while Player 2 is managed by the AI. 

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




