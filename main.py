import numpy as np
import os

from q_learning import Q_learning_agent
from sarsa_lambda import Sarsa_lambda_agent
from DQN.DQN import DQN
from DQN_Masking.DQN import DQN_Masking
from DQN_Masking.DDQN import DDQN_Masking
from DQN_Masking.PER_DQN import PER_DQN_Masking
from DQN_Masking.PER_DDQN import PER_DDQN_Masking
from PPO.PPO import PPO
import gym
from graphs import plot_test_rewards, plot_episode_lengths
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

def manual_reward(folder):
    filepath = os.path.join(folder, 'rewards.txt')

    with open(filepath, 'r') as f:
        contents = f.read()
    parts = contents.split('\n')
    train_rewards = eval(parts[0])
    test_rewards = eval(parts[1])

    print(f"Rewards logs have been loaded from '{filepath}' successfully.")

    window_size = 400
    average_test_rewards = [np.mean(test_rewards[i-window_size:i+1]) if i>window_size else max(0, np.mean(test_rewards[0:i+1])) for i in range(len(test_rewards))]
    average_rewards = [np.mean(train_rewards[i-window_size:i+1]) if i>window_size else np.mean(train_rewards[0:i+1]) for i in range(len(train_rewards))]
    plot_test_rewards(average_rewards, average_test_rewards)

def manual_length(folder):
    filepath = os.path.join(folder, 'rewards.txt')

    with open(filepath, 'r') as f:
        contents = f.read()
    parts = contents.split('\n')
    train_lengths = eval(parts[2])
    test_lengths = eval(parts[3])

    print(f"Length logs have been loaded from '{filepath}' successfully.")

    window_size = 400
    average_test_lengths = [np.mean(test_lengths[i-window_size:i+1]) if i>window_size else np.mean(test_lengths[0:i+1]) for i in range(len(test_lengths))]
    average_train_lengths = [np.mean(train_lengths[i-window_size:i+1]) if i>window_size else np.mean(train_lengths[0:i+1]) for i in range(len(train_lengths))]
    plot_episode_lengths(average_train_lengths, average_test_lengths)

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
epsilon_frame = 20000 # 1,000,000 in DQN paper
trace_decay = 0.7

target_update=200 # 10,000 in DQN paper
batch_size = 32     # from DQN paper
memory_size=100000 # 1,000,000 in DQN paper
learn_interval = 4  # from DQN paper

alpha = 0.6         # from PER paper
beta_start = 0.5    # 0.4 in PER paper
beta_frame = 100000000000
eta = 0.000001

# agent = Q_learning_agent(env, epoch=epoch, lr=0.1, discount=0.99, epsilon=0.15)
# agent = Sarsa_lambda_agent(env, epoch=epoch, lr=0.1, discount=0.99, epsilon=0.15, trace_decay=0.7)
# agent = DQN(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(24,48,96), layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
agent = DQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = DDQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = PER_DQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, beta_frame=beta_frame, eta=eta)
# agent = PER_DDQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, beta_frame=beta_frame, eta=eta)

# agent = PPO(env, epoch=epoch, lr=0.00025, discount=0.99, trace_decay=0.95, eps_clip=0.1, c1=1.0, c2=100.0, channels=(28,56,1), actor_layer_dims=[512,1154,1796], critic_layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=32, learning_interval=100)

# agent.load_training('DDQN_Masking 4p 2024-04-26_22-51-07, 10000 epochs')

agent.train(no_epochs=20000, save=True)

# agent.load_training('DQN_Masking 2024-04-30_10-38-47, 20 epochs')
# agent.show_rewards()
# agent.show_lengths()

# while(True):
#     agent.play_human()

# relative_path = 'PER_DQN_Masking 4p beta aneal 2024-04-20_15-48-43, 10000 epochs'
# manual_reward(relative_path)
# manual_length(relative_path)