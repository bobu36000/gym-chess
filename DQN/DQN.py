import random, time, copy, os
import numpy as np
from datetime import datetime

from gym_chess import ChessEnvV1, ChessEnvV2
from graphs import plot_test_rewards, plot_episode_lengths

from agent import Agent
from DQN.DQN_Network import Network
from ReplayBuffer import ReplayBuffer
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(Agent):
    def __init__(self, environment, epoch, lr, discount, epsilon, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size, learn_interval):
        super().__init__(environment)

        self.name = "DQN"
        self.learn_time = 0
        self.post_move_time = 0
        self.post_next_move_time = 0
        self.rewards = []
        self.test_rewards = []
        self.train_lengths = []
        self.test_lengths = []

        # set hyperparameters
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.target_update = target_update
        self.learn_interval = learn_interval

        self.step = 0
        self.epoch = epoch

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")

        # define CNN
        self.channels = channels
        self.layer_dims = layer_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.q_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction=None).to(self.device)
        self.target_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction=None).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        #define Replay Buffer
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size, batch_size)

        # vectorized functions
        self.slicer = np.vectorize(lambda board: self.slice_board(board), signature='(8, 8)->(14, 8, 8)')
        self.attack_slicer = np.vectorize(lambda state: self.attack_slices(state))
        self.post_move = np.vectorize(lambda state, player, action: self.post_move_state(state, player, action))
        self.preprocess = np.vectorize(lambda state: self.preprocess_state(state))

        self.empty_slice = np.zeros((14,8,8))

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
        next_available_actions = samples["next_available_actions"]
        reward_sample = T.from_numpy(samples["rewards"]).to(self.device)
        terminal_sample = T.from_numpy(samples["terminals"]).to(self.device)
        index_sample = samples["indexes"]

        start = time.time()
        post_move_sample = self.post_move(state_sample, "WHITE", action_sample)
        slice_post_move_sample = T.tensor(np.array([self.preprocess_state(state) for state in post_move_sample])).to(self.device)
        mid= time.time()
        next_action_slices = np.array([self.best_action(next_state_sample[i], next_available_actions[i])[1] if not terminal_sample[i] else self.empty_slice for i in range(len(next_state_sample))])
        slice_post_next_move_sample = T.tensor(next_action_slices).to(self.device)
        self.post_next_move_time += time.time() - mid
        self.post_move_time += mid - start

        # calculate q values
        q_value = self.q_network(slice_post_move_sample.to(dtype=T.float32))[index_sample]
        q_next = self.target_network(slice_post_next_move_sample.to(dtype=T.float32))[index_sample]
        q_next[terminal_sample] = 0.0

        reward_sample = reward_sample.reshape((len(reward_sample),1))
        q_target = reward_sample + self.discount * q_next
        loss = self.q_network.loss(q_target, q_value).to(self.device)
        loss.backward()

        self.q_network.optimizer.step()

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
                # May cause and error if white's move ends the game and then black doesn't play
                self.memory.store(pre_w_state, new_state, white_action, next_available_actions, reward, done)
                
                # if there are enough transitions in the memory for a full batch, then learn (every *learn_interval* time steps)
                if self.memory.full_batch() and self.step%self.learn_interval==0:
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
                    epoch_episode_lengths.append(np.mean(episode_lengths))
                    test_reward, test_length = self.one_episode()
                    test_rewards.append(test_reward)
                    test_lengths.append(test_length)

                    # reset the epoch reward array
                    epoch_reward = []
                    episode_lengths = []

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
        print(f"Time taken by post move section: {round(self.post_move_time, 1)}")
        print(f"Time taken by post next move section: {round(self.post_next_move_time, 1)}")
        print(f"Number of epochs: {no_epochs}")
        print(f"Hyperparameters: lr={self.lr}, discount={self.discount}, epsilon={self.epsilon}, target_update={self.target_update}, learn_interval={self.learn_interval}")
        print(f"Network Parameters: channels={self.channels}, layer_dim={self.layer_dims}, kernel_size={self.kernel_size}, stride={self.stride}, batch_size={self.batch_size}")
        
        if(save):
            self.save_training(no_epochs)

        self.show_rewards(no_epochs)
        self.show_lengths(no_epochs)

    def best_action(self, state, actions):
        board_states = self.post_move(state, "WHITE", actions)
        slices = np.array([self.preprocess_state(state) for state in board_states])

        # calculate values of the states from the q network
        net_out = self.q_network(T.tensor(slices.astype('float32')).to(self.device))

        # find the max value
        best_index = T.argmax(net_out)

        return actions[best_index], slices[best_index]

    def choose_egreedy_action(self, state, actions):
        if(random.random() > self.epsilon):
            # select action with the largest value
            chosen_action, _ = self.best_action(state, actions)
        else:
            # select random action
            chosen_action = random.choice(actions)
        
        return chosen_action

    def one_episode(self):
        environment = ChessEnvV2(player_color=self.env.player, opponent=self.env.opponent, log=False, initial_board=self.env.initial_board, end = self.env.end)
        total_reward = 0
        length = 0

        # iterate through moves
        while(not environment.done):
            length += 1
            available_actions = environment.possible_actions

            action, _ = self.best_action(environment.state, available_actions)
            
            _, w_move_reward, _, _ = environment.white_step(action)
            total_reward += w_move_reward
            if(environment.done):
                break

            available_actions = environment.possible_actions
            
            action = random.choice(available_actions)

            _, black_reward, _, _ = environment.black_step(action)
            total_reward += black_reward
        
        return total_reward, length

    def save_training(self, no_epochs):
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = self.name + " " + date_time + ", " + str(no_epochs) + " epochs"
        self.save_parameters(folder_name, "model.pth")
        self.save_rewards(folder_name, "rewards.txt")
        
    def save_rewards(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w') as f:
            f.write(f"{self.rewards}\n{self.test_rewards}\n{self.train_lengths}\n{self.test_lengths}")
        print(f"Reward logs have been written to '{filepath}' successfully.")

    def save_parameters(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        filepath = os.path.join(folder, filename)

        T.save(self.q_network, filepath)
        print(f"Model successfully written to '{filepath}'")

    def load_training(self, folder):
        # load model
        self.load_parameters(folder, "model.pth")

        # load rewards
        self.load_rewards(folder, "rewards.txt")

    def load_rewards(self, folder, filename):
        # Construct the full file path
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'r') as f:
            contents = f.read()
        parts = contents.split('\n')
        self.rewards = eval(parts[0])
        self.test_rewards = eval(parts[1])
        self.train_lengths = eval(parts[2])
        self.test_lengths = eval(parts[3])

        print(f"Rewards logs have been loaded from '{filepath}' successfully.")

    def load_parameters(self, folder, filename):
        # Construct the full file path
        filepath = os.path.join(folder, filename)

        self.q_network = T.load(filepath)
        self.target_network = T.load(filepath)
        print(f"Model successfully loaded from '{filepath}'")

    def show_rewards(self, no_epochs):
        print("Showing rewards...")
        # calculate rolling averages
        window_size = no_epochs//25
        average_test_rewards = [np.mean(self.test_rewards[i-window_size:i+1]) if i>window_size else max(0, np.mean(self.test_rewards[0:i+1])) for i in range(len(self.test_rewards))]
        average_rewards = [np.mean(self.rewards[i-window_size:i+1]) if i>window_size else np.mean(self.rewards[0:i+1]) for i in range(len(self.rewards))]
        plot_test_rewards(average_rewards, average_test_rewards, self.lr, self.discount, self.epsilon)

    def show_lengths(self, no_epochs):
        # calculate rolling averages
        window_size = no_epochs//25
        average_test_lengths = [np.mean(self.test_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.test_lengths[0:i+1]) for i in range(len(self.test_lengths))]
        average_train_lengths = [np.mean(self.train_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.train_lengths[0:i+1]) for i in range(len(self.train_lengths))]
        plot_episode_lengths(average_train_lengths, average_test_lengths)

    def slice_board(self, board):
        board = np.array(board)
        board_slices = np.zeros((14, 8, 8))

        for i in range(1, 7):
            board_slices[2*i -2][board == i] = 1
            board_slices[2*i -1][board == -i] = 1
            
        return board_slices
    
    def attack_slices(self,state):
        slices = np.zeros((2,8,8))
        squares_under_attack = self.env.get_squares_under_attack(state=state, player="WHITE")
        if(len(squares_under_attack)>0):
            slices[0][squares_under_attack[:, 0], squares_under_attack[:, 1]] = 1

        squares_under_attack = self.env.get_squares_under_attack(state=state, player="BLACK")
        if(len(squares_under_attack)>0):
            slices[1][squares_under_attack[:, 0], squares_under_attack[:, 1]] = 1

        return slices
    
    def preprocess_state(self, state):
        slice = self.slicer(state['board'])
        attack_slices = self.attack_slicer(state)

        slice[12] = attack_slices[0]
        slice[13] = attack_slices[1]

        return slice
    
    def post_move_state(self, state, player, action):
        move = self.env.action_to_move(action)
        next_state, _ = self.env.next_state(state, player, move)
        return next_state
        
    def play_human(self):
        print("Starting Game:")
        self.env.reset()
        total_reward = 0
        
        # iterate through moves
        while(not self.env.done):
            self.env.render()

            available_actions = self.env.possible_actions
            action, _ = self.best_action(self.env.state, available_actions)
            
            _, white_reward, _, _ = self.env.white_step(action)
            self.env.render()
            total_reward += white_reward
            if(self.env.done):
                break

            moves = self.env.possible_moves
            print("Possible moves:")
            for i in range(len(moves)):
                print(i, self.env.move_to_string(moves[i]))
            index = int(input())
            move = moves[index]
            action = self.env.move_to_action(move)

            _, black_reward, _, _ = self.env.black_step(action)
            total_reward += black_reward
        
        self.env.render()
        print(f'Total reward: {total_reward}')