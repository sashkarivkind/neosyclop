# Train an RL agent via double-Q learning to play the game of Nim.
# The agent is trained against a random agent.
# The agent is then tested against a random agent and a minimax agent.
# The agent is then saved to a file.

from nim import Nim
from agent import Agent
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from q_agent import QAgent
from double_q_agent import DoubleQAgent
import numpy as np

# Set the number of training episodes.
num_episodes = 10000

# Set the number of test episodes.
num_test_episodes = 1000

# Set the number of stones in the initial pile.
num_stones = 10

# Set the maximum number of stones that can be removed from the pile.
max_stones = 3

# Set the learning rate.
learning_rate = 0.1

# Set the discount factor.
discount_factor = 0.9

# Set the epsilon-greedy exploration factor.
exploration_factor = 0.1

# Create the game.
game = Nim(num_stones, max_stones)

# Create the agent.
agent = DoubleQAgent(game, learning_rate, discount_factor, exploration_factor)

# Create the random agent.
random_agent = RandomAgent(game)

# Create the minimax agent.
minimax_agent = MinimaxAgent(game)

# Create the Q agent.
q_agent = QAgent(game, learning_rate, discount_factor, exploration_factor)

# Create a list to hold the number of steps per episode.
num_steps_list = []

# Create a list to hold the number of wins per episode.
num_wins_list = []

# Create a list to hold the number of losses per episode.
num_losses_list = []

# Create a list to hold the number of ties per episode.
num_ties_list = []

# Create a list to hold the number of wins against the random agent per episode.
num_random_wins_list = []

# Create a list to hold the number of losses against the random agent per episode.
num_random_losses_list = []

# Create a list to hold the number of ties against the random agent per episode.
num_random_ties_list = []

# Create a list to hold the number of wins against the minimax agent per episode.
num_minimax_wins_list = []

# Create a list to hold the number of losses against the minimax agent per episode.
num_minimax_losses_list = []

# Create a list to hold the number