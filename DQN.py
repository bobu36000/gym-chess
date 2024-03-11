import random, time, copy, os
import numpy as np
from gym_chess import ChessEnvV1, ChessEnvV2
from agent import Agent
from Network import Network


class DQN(Agent):
    def __init__(self, environment, alpha, discount, epsilon, channels, layer_dim, kernel_size, stride):
        super().__init__(environment)

        # set hyperparameters
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon

        # define CNN
        self.network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None)