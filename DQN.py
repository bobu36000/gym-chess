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
    def __init__(self, environment, epoch, alpha, discount, epsilon, target_update, channels, layer_dim, kernel_size, stride, batch_size,  memory_size):
        super().__init__(environment)

        # set hyperparameters
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.target_update = target_update

        self.step = 0
        self.epoch = epoch

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")

        # define CNN
        self.channels = channels
        self.layer_dim = layer_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.q_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None).to(self.device)
        self.target_network = Network(alpha, channels, layer_dim, kernel_size, stride, reduction=None).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        #define Replay Buffer
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size, batch_size)

        self.slicer = np.vectorize(self.slice_board, signature='(8, 8)->(12, 8, 8)')

    def learn(self):
        # print(f"Learning on step {self.step}")
        self.q_network.optimizer.zero_grad()

        if self.step % self.target_update == 0:
            # update the target network's parameters
            self.target_network.load_state_dict(self.q_network.state_dict())

        # sample a batch from memory and extract values
        samples = self.memory.sample_batch()
        state_sample = samples["states"]
        next_state_sample = samples["next_states"]
        action_sample = samples["actions"]
        next_action_sample = samples["next_actions"]
        reward_sample = T.from_numpy(samples["rewards"]).to(self.device)
        terminal_sample = T.from_numpy(samples["terminals"]).to(self.device)
        index_sample = samples["indexes"]

        post_move_sample = np.vectorize(self.post_move_state)(state_sample, "WHITE", action_sample)   #assumes vectorisation is possible with this
        slice_post_move_sample = T.tensor(np.array([self.slicer(state['board']) for state in post_move_sample])).to(self.device)
        post_next_move_sample = np.vectorize(self.post_move_state)(next_state_sample, "WHITE", next_action_sample)
        slice_post_next_move_sample = T.tensor(np.array([self.slicer(state['board']) for state in post_next_move_sample])).to(self.device)

        # calculate q values
        q_value = self.q_network(slice_post_move_sample.to(dtype=T.float32))[index_sample]
        q_next = self.target_network(slice_post_next_move_sample.to(dtype=T.float32))[index_sample]
        q_next[terminal_sample] = 0.0

        reward_sample = reward_sample.reshape((len(reward_sample),1))
        q_target = reward_sample + self.discount * q_next
        loss = self.q_network.loss(q_target, q_value).to(self.device)
        loss.backward()

        self.q_network.optimizer.step()

    def train(self, no_epochs):
        epoch_rewards = []
        test_rewards = []
        episode_lengths = []

        print("Starting Position:")
        self.env.render()
        print("Training")
        start = time.time()

        epoch_reward = []
        # iterate through the number of epochs
        while(self.step/self.epoch < no_epochs):
            episode_reward = 0
            episode_length = 0
            self.env.reset()
            
            # loop through the steps in an episode
            done = False
            while(not done):
                pre_w_state = self.env.state

                #White's move
                available_actions = self.env.possible_actions

                white_action = self.choose_egreedy_action(self.env.state, available_actions)
                new_state, w_move_reward, done, _ = self.env.white_step(white_action)

                if(not done):   # black doesn't play if white's move ended the game
                    if(self.env.opponent == 'self'):
                        temp_state = copy.deepcopy(self.env.state)
                        temp_state['board'] = self.env.reverse_board(self.env.state['board'])

                        possible_moves = self.env.get_possible_moves(state=temp_state, player=self.env.player)
                        available_actions = []
                        for move in possible_moves:
                            available_actions.append(self.env.move_to_action(move))
                        
                        best_action = self.choose_egreedy_action(temp_state, available_actions)

                        # reverse action back to Black's POV
                        black_action = self.env.reverse_action(best_action)

                    elif(self.env.opponent == 'random'):
                        black_actions = self.env.get_possible_actions()
                        black_action = random.choice(black_actions)
                    else:
                        raise ValueError("Invalid opponent type in environment")
                    
                    new_state, b_move_reward, done, _ = self.env.black_step(black_action)

                # store transition
                reward = w_move_reward + b_move_reward
                available_actions = self.env.possible_actions
                if(not done):
                    next_action = self.best_action(new_state, available_actions)
                else:
                    next_action = 4096  # action out of range
                # May cause and error is white's move ends the game and then black doesn't play
                self.memory.store(pre_w_state, new_state, white_action, next_action, reward, done)
                
                # if there are enough transitions in the memory for a full batch, then learn
                if self.memory.full_batch():
                    self.learn()

                episode_reward += reward
                episode_length += 1
                self.step += 1

                # check if it is the end of an epoch
                if(self.step % self.epoch == 0):
                    epoch_rewards.append(np.mean(epoch_reward))
                    test_rewards.append(self.one_episode())

                    # reset the epoch reward array
                    epoch_reward = []

            epoch_reward.append(round(episode_reward, 1))
            episode_lengths.append(episode_length)

        end = time.time()

        # Create an array to store the rolling averages
        average_rewards = np.zeros_like(epoch_rewards, dtype=float)
        average_test_rewards = np.zeros_like(test_rewards, dtype=float)
        
        # calculate rolling averages
        # window_size = no_epochs//25
        window_size = 10
        average_test_rewards = [np.mean(test_rewards[i-window_size:i]) if i>window_size else np.mean(epoch_rewards[0:i+1]) for i in range(len(test_rewards))]
        average_rewards = [np.mean(epoch_rewards[i-window_size:i]) if i>window_size else np.mean(epoch_rewards[0:i+1]) for i in range(len(epoch_rewards))]


        print("Training complete")
        print(f'Time taken: {round(end-start, 1)}')
        print(f"Number of epochs: {no_epochs}")
        print(f"Average episode length: {np.mean(episode_lengths)}")
        print(f"Hyperparameters: alpha={self.alpha}, discount={self.discount}, epsilon={self.epsilon}, target_update={self.target_update}")
        print(f"Network Parameters: channels={self.channels}, layer_dim={self.layer_dim}, kernel_size={self.kernel_size}, stride={self.stride}, batch_size={self.batch_size}")
        
        return(average_rewards, average_test_rewards)

    def best_action(self, state, actions):
        board_states = np.vectorize(self.post_move_state)(state, "WHITE", actions)
        slices = np.array([self.slicer(state['board']) for state in board_states])
        
        # calculate values of the states from the q network
        net_out = self.q_network(T.tensor(slices.astype('float32')).to(self.device))

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

    def one_episode(self):
        environment = ChessEnvV2(player_color=self.env.player, opponent=self.env.opponent, log=False, initial_board=self.env.initial_board, end = self.env.end)
        total_reward = 0

        # iterate through moves
        while(not environment.done):

            available_actions = environment.possible_actions

            action = self.best_action(environment.state, available_actions)
            
            _, w_move_reward, _, _ = environment.white_step(action)
            total_reward += w_move_reward
            if(environment.done):
                break

            available_actions = environment.possible_actions
            
            action = environment.move_to_action(available_actions)

            _, black_reward, _, _ = environment.black_step(action)
            total_reward += black_reward
        
        return total_reward

    def slice_board(self, board):
        board = np.array(board)
        board_slices = np.zeros((12, 8, 8))

        for i in range(1, 7):
            board_slices[2*i -2][board == i] = 1
            board_slices[2*i -1][board == -i] = 1
            
        return board_slices
    
    def post_move_state(self, state, player, action):
        move = self.env.action_to_move(action)
        next_state, _ = self.env.next_state(state, player, move)
        return next_state
        