import random, time, copy, os
import numpy as np
from gym_chess import ChessEnvV1, ChessEnvV2
from agent import Agent
from Network import Network
from ReplayBuffer import ReplayBuffer
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(Agent):
    def __init__(self, environment, alpha, discount, epsilon, target_update, channels, layer_dim, kernel_size, stride, memory_size, batch_size):
        super().__init__(environment)

        # set hyperparameters
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.target_update = target_update

        self.step = 0

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")

        # define CNN
        self.q_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None)
        self.target_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None)

        #define Replay Buffer
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size, batch_size)

    def learn(self):
        self.q_network.optimizer.zero_grad()

        if self.step % self.target_update == 0:
            # update the target network's parameters
            self.target_network.load_state_dict(self.q_network.state_dict())

        # sample a batch from memory and extract values
        samples = self.memory.sample_batch()
        state_sample = T.from_numpy(samples["states"]).to(self.device)
        next_state_sample = T.from_numpy(samples["next_states"]).to(self.device)
        action_sample = samples["actions"]
        next_action_sample = samples["next_actions"]
        reward_sample = T.from_numpy(samples["rewards"]).to(self.device)
        terminal_sample = T.from_numpy(samples["terminals"]).to(self.device)
        index_sample = samples["indexes"]

        post_move_sample = self.post_move_state(state_sample, "WHITE", action_sample)   #assumes vectorisation is possible with this
        post_next_move_sample = self.post_move_state(next_state_sample, "WHITE", next_action_sample)

        # calculate q values
        q_value = self.q_network(self.slice_board(post_move_sample['board']))[index_sample]
        q_next = self.target_network(self.slice_board(post_next_move_sample['board']))[index_sample]
        q_next[terminal_sample] = 0.0
        q_target = reward_sample + self.discount * q_next

        loss = self.q_network.loss(q_target, q_value).to(self.device)
        loss.backward()

        self.q_network.optimizer.step()

    def train(self, no_epochs):
        pass

    def best_action(self, state, actions):
        # note this is assuming vectorisation is happening, it might not work
        board_states = self.post_move_state(state, "WHITE", actions)
        slices = self.slice_board(board_states['board'])
        
        # calculate values of the states from the q network
        net_out = self.q_network(T.tensor(slices))

        # find the max value
        best_index = T.argmax(net_out)

        return actions[best_index]

    def choose_egreedy_action(self, state, actions):
        if(random.random() > self.epsilon):
            # select action with the largest value
            chosen_action = self.best_action(state, actions)
        else:
            # select random action
            chosen_action = random.choice(actions)
        
        return chosen_action

    def slice_board(self, board):
        board = np.array(board)
        board_slices = np.zeros((12, 8, 8))

        for i in range(1, 7):
            board_slices[2*i -2][board == i] = 1
            board_slices[2*i -1][board == -i] = 1
            
        return board_slices
    
    # TODO test this method works with vectorisation of arguments
    def post_move_state(self, state, player, action):
        move = self.env.action_to_move(action)
        next_state, reward = self.env.next_state(state, player, move)
        if(reward!=0):
            raise ValueError("Reward not accounted for")
        return next_state
        