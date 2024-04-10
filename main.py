import numpy as np

from q_learning import Q_learning_agent
from sarsa_lambda import Sarsa_lambda_agent
from DQN.DQN import DQN
from DQN_Masking.DQN import DQN_Masking
from DQN_Masking.DDQN import DDQN_Masking
from DQN_Masking.PER_DQN import PER_DQN_Masking
from DQN_Masking.PER_DDQN import PER_DDQN_Masking
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
PAWN_BOARD[1, 0] = -PAWN_ID
PAWN_BOARD[1, 1] = -PAWN_ID
PAWN_BOARD[1, 2] = -PAWN_ID
PAWN_BOARD[1, 3] = -PAWN_ID
PAWN_BOARD[1, 4] = -PAWN_ID
PAWN_BOARD[1, 5] = -PAWN_ID
PAWN_BOARD[1, 6] = -PAWN_ID
PAWN_BOARD[1, 7] = -PAWN_ID
PAWN_BOARD[6, 0] = PAWN_ID
PAWN_BOARD[6, 1] = PAWN_ID
PAWN_BOARD[6, 2] = PAWN_ID
PAWN_BOARD[6, 3] = PAWN_ID
PAWN_BOARD[6, 4] = PAWN_ID
PAWN_BOARD[6, 5] = PAWN_ID
PAWN_BOARD[6, 6] = PAWN_ID
PAWN_BOARD[6, 7] = PAWN_ID
# PAWN_BOARD[7, 4] = KING_ID
# PAWN_BOARD[0, 4] = -KING_ID


print("Building environment")
env = ChessEnvV2(player_color="WHITE", opponent="self", log=False, initial_board=PAWN_BOARD, end = "promotion")

epoch = 100

lr = 0.00025        # from DQN paper
discount = 0.99     # from DQN paper
epsilon_start = 1.0 # from DQN paper
epsilon_min = 0.1   # from DQN paper
epsilon_frame = 1000000
trace_decay = 0.7

target_update=10000 # from DQN paper
batch_size = 32     # from DQN paper
memory_size=1000000 # from DQN paper
learn_interval = 4  # from DQN paper

alpha = 0.6         # from PER paper
beta_start = 0.4    # from PER paper
beta_max = 1.0      # from PER paper
eta = 0.000001

# agent = Q_learning_agent(env, epoch=epoch, lr=lr, discount=discount, epsilon=epsilon)
# agent = Sarsa_lambda_agent(env, epoch=epoch, lr=lr, discount=discount, epsilon=epsilon, trace_decay=trace_decay)
# agent = DQN(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(24,48,96), layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = DQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = DDQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = PER_DQN_Masking(env, epoch, lr/4, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, eta=eta)
agent = PER_DDQN_Masking(env, epoch, lr/4, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, eta=eta)

# ql_agent.load_q_table('saved_models', 'test_table.txt')

agent.train(no_epochs=10000, save=False)

# agent.save_q_table('saved_models', '4p-Sarsa-10000.txt')
# agent.save_q_table('saved_models', 'test.txt')

# agent.load_training('PER_DDQN_Masking 8p 2024-04-09_21-50-52, 10000 epochs')
# agent.show_rewards()

# while(True):
#     agent.play_human()
