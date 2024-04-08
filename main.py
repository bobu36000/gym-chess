import random, time
import numpy as np
from collections import defaultdict
from graphs import plot_rewards, plot_test_rewards
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from q_learning import Q_learning_agent
from sarsa_lambda import Sarsa_lambda_agent
from DQN.DQN import DQN
from DQN_Masking.DQN import DQN_Masking
from DQN_Masking.DDQN import DDQN_Masking
from DQN_Masking.PER_DQN import PER_DQN_Masking
from DQN_Masking.PER_DDQN import PER_DDQN_Masking
from ReplayBuffer import ReplayBuffer
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
# PAWN_BOARD[6, 6] = PAWN
# PAWN_BOARD[7, 4] = KING_ID
# PAWN_BOARD[0, 4] = -KING_ID


print("Building environment")
#env = ChessEnvV1(player_color="WHITE", opponent="random", log=True, initial_state = PAWN_BOARD, end = "promotion")
env = ChessEnvV2(player_color="WHITE", opponent="self", log=False, initial_board=PAWN_BOARD, end = "promotion")

epoch = 100

lr = 0.00025 #0.02
discount = 0.99 #0.9
epsilon = 0.15
trace_decay = 0.7

target_update=100
batch_size = 32 #100
memory_size=100000 #10000
learn_interval = 4 #10

alpha = 0.6
beta = 0.5
eta = 0.000001

# agent = Q_learning_agent(env, epoch=epoch, lr=lr, discount=discount, epsilon=epsilon)
# agent = Sarsa_lambda_agent(env, epoch=epoch, lr=lr, discount=discount, epsilon=epsilon, trace_decay=trace_decay)
# agent = DQN(env, epoch, lr, discount, epsilon, target_update=target_update, channels=(24,48,96), layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size)
# agent = DQN_Masking(env, epoch, lr, discount, epsilon, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size)
# agent = DDQN_Masking(env, epoch, lr, discount, epsilon, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = PER_DQN_Masking(env, epoch, lr, discount, epsilon, target_update=target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta, eta=eta)
agent = PER_DDQN_Masking(env, epoch, lr, discount, epsilon, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta, eta=eta)

# ql_agent.load_q_table('saved_models', 'test_table.txt')
# agent.load_training('PER_DDQN_Masking 2024-04-08_13-01-41, 200 epochs')
# agent.show_rewards(200)
# agent.show_lengths(200)

agent.train(no_epochs=2000, save=True)

# agent.save_q_table('saved_models', '4p-Sarsa-10000.txt')
# agent.save_q_table('saved_models', 'test.txt')
# agent.save_parameters('saved_models', 'DQN-test.pth')


# while(True):
#     agent.play_human()
