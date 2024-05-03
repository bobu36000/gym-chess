import random, time, copy, os
import numpy as np
from datetime import datetime
from gym_chess import ChessEnvV2
from graphs import plot_test_rewards, plot_multiple_test_rewards
from DQN.DQN import DQN
from DQN_Masking.DQN_Network import Network
from ReplayBuffer import PrioritizedReplayBuffer
# PyTorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PER_DDQN_Masking(DQN):
    def __init__(self, environment, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size, learn_interval, alpha, beta, beta_frame, eta):
        super().__init__(environment, epoch, lr, discount, epsilon_start, epsilon_min, epsilon_frame, target_update, channels, layer_dims, kernel_size, stride, batch_size, memory_size, learn_interval)

        self.name = "PER_DDQN_Masking"

        self.q_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction="none").to(self.device)
        self.target_network = Network(lr, channels, layer_dims, kernel_size, stride, reduction="none").to(self.device)

        self.alpha = alpha
        self.beta = beta
        self.beta_delta = (1.0 - beta) / beta_frame
        self.eta = eta
        self.memory = PrioritizedReplayBuffer(memory_size=memory_size, batch_size=batch_size, alpha=alpha)

        self.no_previous_models = 5
        self.previous_models = []
        self.previous_models.append(copy.deepcopy(self.q_network))
        self.version_rewards = [[]]


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

        #calculate target values
        target_output = self.target_network(slice_next_state_sample.to(dtype=T.float32))
        q_output = self.q_network(slice_next_state_sample.to(dtype=T.float32))

        # mask invalid actions
        q_output_masked = T.zeros_like(target_output) -1000
        q_output_masked[next_action_mask==1] = q_output[next_action_mask==1]
        q_next = target_output[index_sample, q_output_masked.argmax(dim=1)]

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
                reward = w_move_reward

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
                reward += b_move_reward
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
                    epoch = self.step//self.epoch
                    print(f"Epoch: {epoch}")
                    if epoch%(no_epochs//self.no_previous_models) == 0:
                        print("Copying current model...")
                        self.previous_models.append(copy.deepcopy(self.q_network))
                        self.version_rewards.append([])

                    epoch_rewards.append(np.mean(epoch_reward))
                    epoch_episode_lengths.append(np.mean(episode_lengths))
                    test_reward, test_length = self.one_episode()
                    test_rewards.append(test_reward)
                    test_lengths.append(test_length)
                    self.play_previous()

                    # reset the epoch reward array
                    epoch_reward = []
                    episode_lengths = []

                self.epsilon = max(self.epsilon + self.epsilon_delta, self.epsilon_min)
                self.beta = min(self.beta + self.beta_delta, 1.0)

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

        return best_action
    
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
        length = 0

        # iterate through moves
        while(not environment.done):
            length += 1
            available_actions = environment.possible_actions

            action = self.best_action(environment.state, available_actions)
            
            _, w_move_reward, _, _ = environment.white_step(action)
            total_reward += w_move_reward
            if(environment.done):
                break

            available_actions = environment.possible_actions
            
            action = random.choice(available_actions)

            _, black_reward, _, _ = environment.black_step(action)
            total_reward += black_reward
        
        return total_reward, length

    def play_previous(self):
        environment = ChessEnvV2(player_color=self.env.player, opponent=self.env.opponent, log=False, initial_board=self.env.initial_board, end = self.env.end)

        for i in range(len(self.previous_models)):
            total_reward = 0
            environment.reset()

            # iterate through moves
            while(not environment.done):
                available_actions = environment.possible_actions

                action = self.best_action(environment.state, available_actions)
                
                _, w_move_reward, _, _ = environment.white_step(action)
                total_reward += w_move_reward
                if(environment.done):
                    break


                temp_state = copy.deepcopy(environment.state)
                temp_state['board'] = environment.reverse_board(environment.state['board'])

                possible_moves = environment.get_possible_moves(state=temp_state, player=environment.player)
                available_actions = []
                for move in possible_moves:
                    available_actions.append(environment.move_to_action(move))
                
                #### find the best action using the specified model ####
                slice = np.array(self.preprocess_state(temp_state))
                net_out = self.previous_models[i](T.tensor(slice.astype('float32')).to(self.device))[0]
                action_values = net_out[available_actions]
                _, best_index = T.max(action_values, dim=0)
                best_index = best_index.item()
                best_action = available_actions[int(best_index)]
                ####                                                ####

                # reverse action back to Black's POV
                black_action = environment.reverse_action(best_action)

                _, black_reward, _, _ = environment.black_step(black_action)
                total_reward += black_reward
            
            self.version_rewards[i].append(total_reward)

    def save_parameters(self, folder, filename):
        return super().save_parameters(folder, filename)
    
    def save_rewards(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w') as f:
            f.write(f"{self.rewards}\n{self.test_rewards}\n{self.version_rewards}\n{self.train_lengths}\n{self.test_lengths}")
        print(f"Reward logs have been written to '{filepath}' successfully.")

    def load_parameters(self, folder, filename):
        return super().load_parameters(folder, filename)
    
    def load_rewards(self, folder, filename):
        # Construct the full file path
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'r') as f:
            contents = f.read()
        parts = contents.split('\n')
        self.rewards = eval(parts[0])
        self.test_rewards = eval(parts[1])
        self.version_rewards = eval(parts[2])
        self.train_lengths = eval(parts[3])
        self.test_lengths = eval(parts[4])

        print(f"Rewards logs have been loaded from '{filepath}' successfully.")

    def slice_board(self, board):
        return super().slice_board(board)
    
    def preprocess_state(self, state):
        return super().preprocess_state(state)
    
    def play_human(self):
        return super().play_human()
    
    def save_training(self):
        no_epochs = len(self.test_rewards)
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = self.name + " " + date_time + ", " + str(no_epochs) + " epochs"
        self.save_parameters(folder_name, "model.pth")
        self.save_rewards(folder_name, "rewards.txt")

    def show_rewards(self):
        print("Showing rewards...")
        average_test_rewards = []
        no_epochs = len(self.test_rewards)
        # calculate rolling averages
        window_size = 400
        average_test_rewards = [np.mean(self.test_rewards[i-window_size:i+1]) if i>window_size else max(0, np.mean(self.test_rewards[0:i+1])) for i in range(len(self.test_rewards))]
        average_rewards = [np.mean(self.rewards[i-window_size:i+1]) if i>window_size else np.mean(self.rewards[0:i+1]) for i in range(len(self.rewards))]
        plot_test_rewards(average_rewards, average_test_rewards)

        window_size = no_epochs//10
        average_version_rewards = []
        for j in range(len(self.version_rewards)):
            averages = [np.mean(self.version_rewards[j][i-window_size:i]) if i>window_size else max(-20.0, np.mean(self.version_rewards[j][0:i+1])*i/window_size) for i in range(len(self.version_rewards[j]))]
            
            average_version_rewards.append(averages)
        plot_multiple_test_rewards(average_version_rewards)
