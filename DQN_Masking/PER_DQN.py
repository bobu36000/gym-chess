import random, time, copy
import numpy as np
from DQN.DQN import DQN
from DQN_Masking.DQN_Network import Network
from ReplayBuffer import PrioritizedReplayBuffer
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PER_DQN_Masking(DQN):
    def __init__(self, environment, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size, learn_interval, alpha, beta, eta):
        super().__init__(environment, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size, learn_interval)

        self.name = "PER_DQN_Masking"

        self.q_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction="none").to(self.device)
        self.target_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction="none").to(self.device)

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.memory = PrioritizedReplayBuffer(memory_size=memory_size, batch_size=batch_size, alpha=alpha)

    def learn(self):
        self.q_network.optimizer.zero_grad()

        if self.step % self.target_update == 0:
            # update the target network's parameters
            self.target_network.load_state_dict(self.q_network.state_dict())

        # sample a batch from memory and extract values
        samples = self.memory.sample_batch(self.beta)
        state_sample = samples["states"]
        next_state_sample = samples["next_states"]
        action_sample = samples["actions"]
        next_action_mask = samples["next_available_actions"]
        reward_sample = T.from_numpy(samples["rewards"]).to(self.device)
        terminal_sample = T.from_numpy(samples["terminals"]).to(self.device)
        weight_sample = T.from_numpy(samples["weights"]).to(self.device)
        batch_sample = samples["batches"]
        index_sample = samples["indexes"]

        # preprocess states
        slice_state_sample = T.tensor(np.array([self.preprocess_state(state) for state in state_sample])).to(self.device)
        slice_next_state_sample = T.tensor(np.array([self.preprocess_state(state) for state in next_state_sample])).to(self.device)

        # calculate q values
        q_value = self.q_network(slice_state_sample.to(dtype=T.float32))[index_sample, action_sample]

        target_output = self.target_network(slice_next_state_sample.to(dtype=T.float32))
        # mask invalid actions
        q_next = T.zeros_like(target_output) -1000
        q_next[next_action_mask==1] = target_output[next_action_mask==1]
        q_next = q_next.max(dim=1)[0]
        q_next[terminal_sample] = 0.0
        q_target = reward_sample + self.discount * q_next

        loss_sample = self.q_network.loss(q_target, q_value).to(self.device)
        loss = T.mean(loss_sample * weight_sample)
        loss.backward()     

        self.q_network.optimizer.step()

        loss_sample_2 = loss_sample.cpu().detach().numpy()
        loss_sample_3 = loss_sample_2 + self.eta
        self.memory.update_batch(batch_sample, loss_sample_3)
    
    def train(self, no_epochs, save=False):
        epoch_rewards = []
        episode_lengths = []
        epoch_episode_lengths = []
        test_rewards = []
        test_lengths = []

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
                
                # if there are enough transitions in the memory for a full batch, then learn
                if self.memory.full_batch() and self.step%self.learn_interval==0:
                    before_learn = time.time()
                    self.learn()
                    self.learn_time += time.time()-before_learn

                episode_reward += reward
                episode_length += 1
                self.step += 1

                # check if it is the end of an epoch
                if(self.step % self.epoch == 0):
                    print(f"Epoch: {self.step//self.epoch}")
                    epoch_rewards.append(np.mean(epoch_reward))
                    epoch_episode_lengths.append(np.mean(episode_lengths))
                    test_reward, test_length = self.one_episode()
                    test_rewards.append(test_reward)
                    test_lengths.append(test_length)

                    # reset the epoch reward array
                    epoch_reward = []
                    episode_lengths = []

                self.epsilon = max(self.epsilon + self.epsilon_delta, self.epsilon_min)

            epoch_reward.append(round(episode_reward, 1))
            episode_lengths.append(episode_length)

        end = time.time()
        
        self.rewards = epoch_rewards
        self.test_rewards  = test_rewards
        self.train_lengths = epoch_episode_lengths
        self.test_lengths = test_lengths

        print("Training complete")
        print(f'Time taken: {round(end-start, 1)}')
        print(f"Time taken by learn() function: {round(self.learn_time, 1)}")
        print(f"Number of epochs: {no_epochs}")
        print(f"Hyperparameters: lr={self.lr}, discount={self.discount}, epsilon={self.epsilon}, target_update={self.target_update}, learn_interval={self.learn_interval}")
        print(f"Network Parameters: channels={self.channels}, layer_dims={self.layer_dims}, kernel_size={self.kernel_size}, stride={self.stride}, batch_size={self.batch_size}")

        if(save):
            self.save_training()

        self.show_rewards()
        self.show_lengths()

    
    def best_action(self, state, actions):
        slice = np.array(self.preprocess_state(state))

        # calculate values of the states from the q network
        net_out = self.q_network(T.tensor(slice.astype('float32')).to(self.device))[0]  # assumes only one slice is passed in
        
        action_values = net_out[actions]

        # find the max value
        _, best_index = T.max(action_values, dim=0)

        best_action = actions[best_index.item()]

        # self.env.render_grid(grid=self.env.board_to_grid(board=state['board']))
        # for i in range(len(actions)):
        #     print(f"{self.env.action_to_move(actions[i])}: {action_values[i]}")

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
    
    def play_human(self):
        return super().play_human()