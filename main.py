import random, time
import numpy as np
from collections import defaultdict
from graphs import plot_rewards, plot_test_rewards
from q_learning import Q_learning_agent
from sarsa_lambda import Sarsa_lambda_agent
# from q_learning_just_states import q_learning_agent
import gym
from gym_chess import ChessEnvV1, ChessEnvV2
from gym_chess.envs.chess_v1 import (
    KING_ID,
    QUEEN_ID,
    ROOK_ID,
    BISHOP_ID,
    KNIGHT_ID,
    PAWN_ID,
)

def play_human(env):
    total_reward = 0
    #play one episode
    while not env.done:
        env.render()
        #show possible moves
        moves = env.possible_moves
        print("Possible moves:")
        for i in range(len(moves)):
            print(i, env.move_to_string(moves[i]))
        index = int(input())
        move = moves[index]
        action = env.move_to_action(move)

        # pass it to the env and get the next state
        new_state, reward, done, info = env.step(action)
        print(f'Reward: {reward}')
        total_reward += reward
    
    env.render()
    print(f'Total reward: {total_reward}')

DEFAULT_BOARD = [
    [-3, -5, -4, -2, -1, -4, -5, -3],
    [-6, -6, -6, -6, -6, -6, -6, -6],
    [0] * 8,
    [0] * 8,
    [0] * 8,
    [0] * 8,
    [6, 6, 6, 6, 6, 6, 6, 6],
    [3, 5, 4, 2, 1, 4, 5, 3],
]

PAWN_BOARD = np.array([[0] * 8] * 8, dtype=np.int8)
# PAWN_BOARD[1, 0] = -PAWN_ID
# PAWN_BOARD[1, 1] = -PAWN_ID
PAWN_BOARD[1, 2] = -PAWN_ID
PAWN_BOARD[1, 3] = -PAWN_ID
PAWN_BOARD[1, 4] = -PAWN_ID
PAWN_BOARD[1, 5] = -PAWN_ID
# PAWN_BOARD[1, 6] = -PAWN_ID
# PAWN_BOARD[1, 7] = -PAWN_ID
# PAWN_BOARD[6, 0] = PAWN_ID
# PAWN_BOARD[6, 1] = PAWN_ID
PAWN_BOARD[6, 2] = PAWN_ID
PAWN_BOARD[6, 3] = PAWN_ID
PAWN_BOARD[6, 4] = PAWN_ID
PAWN_BOARD[6, 5] = PAWN_ID
# PAWN_BOARD[6, 6] = PAWN_ID
# PAWN_BOARD[6, 7] = PAWN_ID
# PAWN_BOARD[7, 4] = KING_ID
# PAWN_BOARD[0, 4] = -KING_ID


print("Building environment")
#env = ChessEnvV1(player_color="WHITE", opponent="random", log=True, initial_state = PAWN_BOARD, end = "promotion")
env = ChessEnvV2(player_color="WHITE", opponent="self", log=False, initial_board=PAWN_BOARD, end = "promotion")

epoch = 100

alpha = 0.2
discount = 0.9
epsilon = 0.15
trace_decay = 0.7


agent = Sarsa_lambda_agent(env, epoch=epoch, alpha=alpha, discount=discount, epsilon=epsilon, trace_decay=trace_decay)
# agent = Q_learning_agent(env, epoch=epoch, alpha=alpha, discount=discount, epsilon=epsilon)

# ql_agent.load_q_table('saved_tables', 'test_table.txt')

average_rewards, test_rewards = agent.train(no_epochs=10000)

# agent.save_q_table('saved_tables', '4p-Sarsa-10000.txt')
# agent.save_q_table('saved_tables', 'test.txt')

# plot_rewards(average_rewards, alpha, discount, epsilon, goal=100)
plot_test_rewards(average_rewards, test_rewards, alpha, discount, epsilon, goal=100)

# print(agent.get_0_proportion())

# while(True):
#     ql_agent.play_human()