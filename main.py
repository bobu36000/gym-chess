import numpy as np
import random, copy

from q_learning import Q_learning_agent
from sarsa_lambda import Sarsa_lambda_agent
from DQN.DQN import DQN
from DQN_Masking.DQN import DQN_Masking
from DQN_Masking.DDQN import DDQN_Masking
from DQN_Masking.PER_DQN import PER_DQN_Masking
from DQN_Masking.PER_DDQN import PER_DDQN_Masking
from PPO.PPO import PPO
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

def compete(agent1, agent2, env, iterations=100):

    def play_episode(w_agent, b_agent, env):
        env.reset()
        environment = env
        total_reward = 0
        first_move = True

        # iterate through moves
        while(not environment.done):
            available_actions = environment.possible_actions

            if(first_move):
                w_action = random.choice(available_actions)
                first_move = False
            else:
                w_action = w_agent.best_action(environment.state, available_actions)
            
            _, w_move_reward, _, _ = environment.white_step(w_action)
            total_reward += w_move_reward
            if(environment.done):
                break

            
            temp_state = copy.deepcopy(environment.state)
            temp_state['board'] = environment.reverse_board(environment.state['board'])

            possible_moves = environment.get_possible_moves(state=temp_state, player=environment.player)
            available_actions = []
            for move in possible_moves:
                available_actions.append(environment.move_to_action(move))
            action = b_agent.best_action(temp_state, available_actions)
            b_action = environment.reverse_action(action)

            _, black_reward, _, _ = environment.black_step(b_action)
            total_reward += black_reward
        
        if(total_reward) > 50:
            # print("White wins")
            return np.array([1,0,0])
        elif(total_reward < -50):
            # print("Black wins")
            return np.array([0,0,1])
        else:
            # print("Draw")
            return np.array([0,1,0])

    fp_results = np.array([0,0,0])
    sp_results = np.array([0,0,0])
    print("Starting first pass")
    for i in range(iterations//2):
        fp_results += play_episode(agent1, agent2, env)
    print(f"First Pass Results: {fp_results}")
    print("Starting second pass")
    for i in range(iterations//2):
        sp_results += np.flip(play_episode(agent2, agent1, env))
    
    print(f"Second Pass Results: {sp_results}")
    print(f"Total Results: {fp_results+sp_results}")

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
PAWN_BOARD[7, 4] = KING_ID
PAWN_BOARD[0, 4] = -KING_ID


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
beta_start = 0.4    # from PER paper
beta_frame = 100000000000
eta = 0.000001

# agent = Q_learning_agent(env, epoch=epoch, lr=0.1, discount=0.99, epsilon=0.15)
# agent = Sarsa_lambda_agent(env, epoch=epoch, lr=0.1, discount=0.99, epsilon=0.15, trace_decay=0.7)
# agent = DQN(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(24,48,96), layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = DQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = DDQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update=target_update, channels=(28,56,112), layer_dims=[512,1024,2048], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval)
# agent = PER_DQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, beta_frame=beta_frame, eta=eta)
# agent = PER_DDQN_Masking(env, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels=(28,56,1), layer_dims=[512,1154,1796], kernel_size=3, stride=1, batch_size=batch_size, memory_size=memory_size, learn_interval=learn_interval, alpha=alpha, beta=beta_start, beta_frame=beta_frame, eta=eta)

agent = PPO(env, epoch=epoch, lr=0.00025, discount=0.99, trace_decay=0.95, eps_clip=0.1, c1=1.0, c2=100.0, channels=(28,56,1), actor_layer_dims=[512,1154,1796], critic_layer_dims=[128,128,128], kernel_size=3, stride=1, batch_size=32, learning_interval=100)

agent.train(no_epochs=20000, save=True)

# agent.load_training('PPO Kp 2024-05-02_04-07-54, 20000 epochs')
# agent.show_rewards()
# agent.show_lengths()
