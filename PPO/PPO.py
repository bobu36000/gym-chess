import random, time, copy, os
import numpy as np
from datetime import datetime
from gym_chess import ChessEnvV2
from graphs import plot_test_rewards, plot_episode_lengths, plot_multiple_test_rewards
from agent import Agent
from PPO.Critic_Network import Network as CriticNetwork
from PPO.Actor_Network import Network as ActorNetwork
from PPO.Memory import TradjectoryMemory
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PPO(Agent):
    def __init__(self, env, epoch, lr, discount, trace_decay, eps_clip, c1, c2, channels, actor_layer_dims, critic_layer_dims, kernel_size, stride, batch_size, learning_interval):
        super().__init__(env)

        self.name = "PPO"
        self.learn_time = 0
        self.post_move_time = 0
        self.post_next_move_time = 0
        self.rewards = []
        self.test_rewards = []
        self.train_lengths = []
        self.test_lengths = []
        
        self.step = 0
        self.epoch = epoch
        self.learning_interval = learning_interval

        self.lr = lr
        self.discount = discount
        self.trace_decay = trace_decay
        self.eps_clip = eps_clip
        self.c1 = c1
        self.c2 = c2

        self.channels = channels
        self.actor_layer_dims = actor_layer_dims
        self.critic_layer_dims = critic_layer_dims
        self.kernel_size = kernel_size
        self.stride = stride

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Device: {self.device}")
        
        self.critic = CriticNetwork(lr, channels, critic_layer_dims, kernel_size, stride).to(self.device)
        self.actor = ActorNetwork(lr, channels, actor_layer_dims, kernel_size, stride).to(self.device)

        self.memory = TradjectoryMemory(batch_size)

        self.learn_time = 0

        self.no_previous_models = 5
        self.previous_models = []
        self.previous_models.append(copy.deepcopy(self.actor))
        self.version_rewards = [[]]

    def learn(self):
        state_memory, action_memory, action_mask_memory, reward_memory, value_memory, prob_memory, terminal_memory = self.memory.get_memory()
        slice_memory = np.array([self.preprocess_state(state) for state in state_memory])
        
        def delta(t, terminal=False):
                if terminal:
                    return reward_memory[t] - value_memory[t]
                else:
                    return reward_memory[t] + self.discount * value_memory[t+1] - value_memory[t]

        episodes = [[]]
        episode_count = 1
        for i in range(len(reward_memory)):
            episodes[-1].append(i)
            if(terminal_memory[i]):
                episodes.append([])
                episode_count += 1

        # calculate advantages on each episode
        advantages = []
        for i in range(len(episodes)):
            episode_advantages = []
            adv = 0

            T = len(episodes[i])

            for t in range(T-1, -1, -1):
                if t == T-1 or terminal_memory[episodes[i][t]]:
                    adv += delta(episodes[i][t], terminal=True)
                else:
                    adv += (self.discount * self.trace_decay) ** (T - t + 1) * delta(episodes[i][t])
                episode_advantages.append(adv)

            episode_advantages.reverse()
            advantages.append(episode_advantages)
        
        # flatten the advantages list
        advantages = [advantage for episode in advantages for advantage in episode]
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for i in range(3):
            batches = self.memory.get_batches()

            values = torch.tensor(value_memory, dtype=torch.float32).to(self.device)
            for batch in batches:
                slices = slice_memory[batch]
                state_slices = torch.tensor(slices).to(self.device).to(dtype=torch.float32)
                old_probs = torch.tensor(prob_memory[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_memory[batch], dtype=torch.float32).to(self.device)
                action_mask = torch.tensor(action_mask_memory[batch], dtype=torch.float32).to(self.device)

                dist =self.actor(state_slices, action_mask)
                critic_value = self.critic(state_slices)

                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(new_probs - old_probs)
                clipped_probs = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip)

                L_clip = -torch.min(clipped_probs * advantages[batch], prob_ratio * advantages[batch]).mean()

                L_VF = ((advantages[batch] + values[batch] - critic_value)**2).mean()

                loss = L_clip + self.c1 * L_VF - dist.entropy().mean() * self.c2
                # print(f"Clip={L_clip}, L_VF={L_VF}, entropy={dist.entropy().mean()}")

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            # self.actor.scheduler.step()
            # self.critic.scheduler.step()
        self.c2 = max(1.0, self.c2 * 0.999)
        self.memory.clear()

    def train(self, no_epochs, save=False):
        epoch_rewards = []
        episode_lengths = []
        epoch_episode_lengths = []
        test_rewards = []
        test_lengths = []
        epoch_reward = []
        
        print("Starting Position:")
        self.env.render()
        print("Training...")
        start = time.time()

        while(self.step < no_epochs*self.epoch):
            episode_reward = 0
            episode_length = 0
            self.env.reset()

            done = False
            while(not done):
                pre_w_state = self.env.state
                available_actions = self.env.possible_actions
                white_action_mask = np.zeros(4100)
                white_action_mask[available_actions] = 1

                white_action, log_prob, value = self.choose_action(pre_w_state, available_actions)

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
                        
                        best_action, _, _ = self.choose_action(temp_state, available_actions)

                        # reverse action back to Black's POV
                        black_action = self.env.reverse_action(best_action)

                    elif(self.env.opponent == 'random'):
                        black_actions = self.env.get_possible_actions()
                        black_action = random.choice(black_actions)
                    else:
                        raise ValueError("Invalid opponent type in environment")
                    
                    new_state, b_move_reward, done, _ = self.env.black_step(black_action)

                    reward += b_move_reward

                self.memory.store(pre_w_state, white_action, list(white_action_mask), reward, value, log_prob, done)

                if self.step % self.learning_interval == 0 and self.memory.full_batch():
                    start_learn = time.time()
                    self.learn()
                    end_learn = time.time()
                    self.learn_time += end_learn-start_learn

                episode_reward += reward
                episode_length += 1
                self.step += 1

                # check if it is the end of an epoch
                if(self.step % self.epoch == 0):
                    epoch = self.step//self.epoch
                    print(f"Epoch: {epoch}")
                    if epoch%(no_epochs//self.no_previous_models) == 0:
                        print("Copying current model...")
                        self.previous_models.append(copy.deepcopy(self.actor))
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
        print(f"Hyperparameters: lr={self.lr}, discount={self.discount}, trace_decay={self.trace_decay}, eps_clip={self.eps_clip}, c1={self.c1}, c2={self.c2}")
        print(f"Network Parameters: channels={self.channels}, actor_layer_dims={self.actor_layer_dims}, critic_layer_dims={self.critic_layer_dims}, kernel_size={self.kernel_size}, stride={self.stride}")

        if(save):
            self.save_training()

        self.show_rewards()
        self.show_lengths()


    def choose_action(self, state, actions):
        slice = torch.tensor(self.preprocess_state(state).astype('float32')).to(self.device)

        action_mask = torch.zeros(1,4100)
        action_mask[:,actions] = 1

        distribution = self.actor(slice, action_mask)
        # print(f"Entropy={distribution.entropy().mean()}")

        value = self.critic(slice)
        action = distribution.sample()

        probs = distribution.log_prob(action).item()
        action = action.item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def one_episode(self):
        environment = ChessEnvV2(player_color=self.env.player, opponent=self.env.opponent, log=False, initial_board=self.env.initial_board, end = self.env.end)
        total_reward = 0
        length = 0

        # iterate through moves
        while(not environment.done):
            length += 1
            available_actions = environment.possible_actions

            action, _, _ = self.choose_action(environment.state, available_actions)
            
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

                action, _ , _= self.choose_action(environment.state, available_actions)
                
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
                
                action_mask = torch.zeros(1,4100)
                action_mask[:,available_actions] = 1
                
                #### find the best action using the specified model ####
                slice = np.array(self.preprocess_state(temp_state))
                distribution = self.previous_models[i](torch.tensor(slice.astype('float32')).to(self.device), action_mask)
                best_action = distribution.sample().item()
                ####                                                ####

                # reverse action back to Black's POV
                black_action = environment.reverse_action(best_action)

                _, black_reward, _, _ = environment.black_step(black_action)
                total_reward += black_reward
            
            self.version_rewards[i].append(total_reward)

    def play_human(self):
        print("Starting Game:")
        self.env.reset()
        total_reward = 0
        
        # iterate through moves
        while(not self.env.done):
            self.env.render()

            available_actions = self.env.possible_actions
            action, probs, value = self.choose_action(self.env.state, available_actions)
            print(f"Probability of chose action = {np.exp(probs)}, value of state = {value}")
            
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

    def preprocess_state(self, state):
        slice = self.slice_board(state['board'])
        attack_slices = self.attack_slices(state)

        slice[12] = attack_slices[0]
        slice[13] = attack_slices[1]

        return slice
    
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
        
    def show_rewards(self):
        print("Showing rewards...")
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

    def show_lengths(self):
        print("Showing lengths...")
        no_epochs = len(self.test_rewards)
        # calculate rolling averages
        window_size = no_epochs//25
        average_test_lengths = [np.mean(self.test_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.test_lengths[0:i+1]) for i in range(len(self.test_lengths))]
        average_train_lengths = [np.mean(self.train_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.train_lengths[0:i+1]) for i in range(len(self.train_lengths))]
        plot_episode_lengths(average_train_lengths, average_test_lengths)

    def save_training(self):
        no_epochs = len(self.test_rewards)
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
            f.write(f"{self.rewards}\n{self.test_rewards}\n{self.version_rewards}\n{self.train_lengths}\n{self.test_lengths}")
        print(f"Reward logs have been written to '{filepath}' successfully.")

    def save_parameters(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        actor_filepath = os.path.join(folder, "actor"+filename)
        critic_filepath = os.path.join(folder, "critic"+filename)

        torch.save(self.actor, actor_filepath)
        torch.save(self.critic, critic_filepath)
        print(f"Model successfully saved")

    def load_training(self, folder):
        # load model
        self.load_parameters(folder, "actormodel.pth", "criticmodel.pth")

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
        self.version_rewards = eval(parts[2])
        self.train_lengths = eval(parts[3])
        self.test_lengths = eval(parts[4])

        print(f"Rewards logs have been loaded from '{filepath}' successfully.")

    def load_parameters(self, folder, actor_filename, critic_filename):
        # Construct the full file path
        actor_filepath = os.path.join(folder, actor_filename)
        self.actor = torch.load(actor_filepath)

        critic_filepath = os.path.join(folder, critic_filename)
        self.target_network = torch.load(critic_filepath)
        print(f"Model successfully loaded from '{critic_filepath}'")