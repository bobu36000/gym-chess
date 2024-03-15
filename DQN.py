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

        #self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")

        # define CNN
        self.q_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None)
        self.target_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None)

        #define Replay Buffer
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size, batch_size)

    def learn(self):
        pass

    def train(self, no_epochs):
        pass

    def best_action(self, state, actions):
        pass

    def choose_egreedy_action(self, state, actions):
        pass

    def slice_board(self, board):
        board = np.array(board)
        board_slices = np.zeros((12, 8, 8))

        for i in range(1, 7):
            board_slices[2*i -2][board == i] = 1
            board_slices[2*i -1][board == -i] = 1
            
        return board_slices
        