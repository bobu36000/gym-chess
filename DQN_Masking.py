import random, time, copy, os
import numpy as np
from gym_chess import ChessEnvV1, ChessEnvV2
from agent import Agent
from DQN import DQN
from DQN_Masking_Network import Network
from ReplayBuffer import ReplayBuffer
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN_Masking(DQN):
    def __init__(self, environment, epoch, lr, discount, epsilon, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size):
        super().__init__(environment, epoch, lr, discount, epsilon, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size)

        self.q_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction=None).to(self.device)
        self.target_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction=None).to(self.device)

    def learn(self):
        self.q_network.optimizer.zero_grad()

        if self.step % self.target_update == 0:
            # update the target network's parameters
            self.target_network.load_state_dict(self.q_network.state_dict())

        # sample a batch from memory and extract values
        samples = self.memory.sample_batch()
        state_sample = samples["states"]
        next_state_sample = samples["next_states"]
        action_sample = samples["actions"]
        next_action_mask = np.vstack(samples["next_available_actions"])
        reward_sample = T.from_numpy(samples["rewards"]).to(self.device)
        terminal_sample = T.from_numpy(samples["terminals"]).to(self.device)
        index_sample = samples["indexes"]

        # preprocess states
        slice_state_sample = T.tensor(np.array([self.preprocess_state(state) for state in state_sample])).to(self.device)
        slice_next_state_sample = T.tensor(np.array([self.preprocess_state(state) for state in next_state_sample])).to(self.device)

        # calculate q values
        q_output = self.q_network(slice_state_sample.to(dtype=T.float32))[index_sample]
        target_output = self.target_network(slice_next_state_sample.to(dtype=T.float32))[index_sample]
        target_output[terminal_sample] = 0.0

        # mask invalid actions
        q_value = T.zeros_like(q_output)
        q_value[next_action_mask==1] = q_output[next_action_mask==1]
        q_next = T.zeros_like(target_output)
        q_next[next_action_mask==1] = target_output[next_action_mask==1]

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
        print("Training...")
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
                next_available_actions = np.array(self.env.possible_actions)
                # create next action mask
                next_action_mask = np.zeros((4100))
                if(len(next_available_actions>0)):
                    next_action_mask[next_available_actions] = 1
                self.memory.store(pre_w_state, new_state, white_action, next_action_mask, reward, done)
                
                # if there are enough transitions in the memory for a full batch, then learn (every 10 time steps)
                if self.memory.full_batch() and self.step%10==0:
                    before_learn = time.time()
                    self.learn()
                    after_learn = time.time()
                    self.learn_time += after_learn-before_learn

                episode_reward += reward
                episode_length += 1
                self.step += 1

                # check if it is the end of an epoch
                if(self.step % self.epoch == 0):
                    print(f"Epoch: {self.step//self.epoch}")
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
        window_size = no_epochs//25
        average_test_rewards = [np.mean(test_rewards[i-window_size:i]) if i>window_size else np.mean(epoch_rewards[0:i+1]) for i in range(len(test_rewards))]
        average_rewards = [np.mean(epoch_rewards[i-window_size:i]) if i>window_size else np.mean(epoch_rewards[0:i+1]) for i in range(len(epoch_rewards))]


        print("Training complete")
        print(f'Time taken: {round(end-start, 1)}')
        print(f"Time taken by learn() function: {round(self.learn_time, 1)}")
        print(f"Number of epochs: {no_epochs}")
        print(f"Average episode length: {np.mean(episode_lengths)}")
        print(f"Hyperparameters: lr={self.lr}, discount={self.discount}, epsilon={self.epsilon}, target_update={self.target_update}")
        print(f"Network Parameters: channels={self.channels}, layer_dim={self.layer_dim}, kernel_size={self.kernel_size}, stride={self.stride}, batch_size={self.batch_size}")
        
        return(average_rewards, average_test_rewards)
    
    def best_action(self, state, actions):
        slice = np.array(self.preprocess_state(state))

        # calculate values of the states from the q network
        net_out = self.q_network(T.tensor(slice.astype('float32')).to(self.device))[0]  # assumes only one slice is passed in
        
        action_values = net_out[actions]

        # find the max value
        _, best_index = T.max(action_values, dim=0)

        best_action = actions[best_index.item()]
        return best_action, None
    
    def choose_egreedy_action(self, state, actions):
        return super().choose_egreedy_action(state, actions)
    
    def one_episode(self):
        return super().one_episode()
    
    def save_parameters(self, folder, filename):
        return super().save_parameters(folder, filename)
    
    def load_parameters(self, folder, filename):
        return super().load_parameters(folder, filename)
    
    def slice_board(self, board):
        return super().slice_board(board)
    
    def preprocess_state(self, state):
        return super().preprocess_state(state)