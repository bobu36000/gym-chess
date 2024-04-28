import random, time, copy, os
import numpy as np
from collections import defaultdict
import gym
from gym_chess import ChessEnvV1, ChessEnvV2
from agent import Agent
from graphs import plot_test_rewards, plot_episode_lengths
from datetime import datetime

class Q_learning_agent(Agent):
    def __init__(self, environment, epoch = 100, lr=0.2, discount=1.0, epsilon=0.15):
        super().__init__(environment)

        self.name = "Q-learning"

        # set hyperparameters
        self.lr = lr          #learning rate
        self.discount = discount    #discount factor
        self.epsilon = epsilon      #epsilon greedy term

        # set how many time steps are in an epoch
        self.epoch = epoch

        # create value lookup table
        self.Q = defaultdict(lambda: 0.0)
        self.default_value = 0.0

        self.test_rewards = []
        self.train_rewards = []

        self.step = 0

    def reset_memory(self):
        # resets the value lookup table
        self.Q = {}

    def train(self, no_epochs=0, save=False, double=False):
        epoch_rewards = []
        episode_lengths = []
        epoch_episode_lengths = []
        test_rewards = []
        test_lengths = []

        print('Starting position:')
        self.env.render()
        print("Training...")
        start = time.time()

        epoch_reward = []

        # loop for each episode
        while(self.step/self.epoch < no_epochs):
            total_reward = 0    # total reward collected over this episode
            self.env.reset()

            episode_length = 0
            pre_b_state = None

            # loop for each action in an episode
            done = False
            while(not done):
                self.step += 1
                pre_w_state = self.env.encode_state()

                # White's move
                available_actions = self.env.possible_actions
                # make sure all actions are initialized in the lookup table

                white_action = self.choose_egreedy_action(pre_w_state, available_actions)
                new_state, w_move_reward, done, info = self.env.white_step(white_action)

                if(pre_b_state != None and self.env.opponent == "self" and double):
                    #updates the table from black's persepctive
                    converted_state = self.env.reverse_state_encoding(state=pre_b_state)
                    converted_action = self.env.reverse_action(black_action)
                    black_reward = -(w_move_reward+black_move_reward)
                    if(done):
                        # when the new state is terminal there are no further actions from it and its Q value is 0
                        self.update_table(converted_state, converted_action, black_reward)
                    else:
                        post_w_state = self.env.encode_state()
                        reversed_post_w_state = self.env.reverse_state_encoding(state=post_w_state)
                        #make sure all actions are initialised in the lookup table
                        available_actions = np.array(self.env.possible_actions)
                        r_available_actions = self.env.reverse_action(available_actions)
                        
                        best_action = self.best_action(reversed_post_w_state, r_available_actions)
                        # self.env.show_encoded_state(reversed_post_w_state)
                        # print(self.env.action_to_move(best_action), [self.env.action_to_move(action) for action in r_available_actions])
                        self.update_table(converted_state, converted_action, black_reward, reversed_post_w_state, best_action)


                if(not done):   # if white's move ended the game, black does not move
                    # Black's move
                    pre_b_state = self.env.encode_state()

                    if(self.env.opponent == 'self'):
                        temp_state = copy.deepcopy(self.env.state)
                        temp_state['board'] = self.env.reverse_board(self.env.state['board'])

                        possible_moves = self.env.get_possible_moves(state=temp_state, player=self.env.player)
                        encoded_temp_state = self.env.encode_state(temp_state)
                        available_actions = []
                        for move in possible_moves:
                            available_actions.append(self.env.move_to_action(move))
                        # make sure all actions are initialized in the lookup table
                        
                        # best_action = self.best_action(encoded_temp_state, possible_actions)
                        best_action = self.choose_egreedy_action(encoded_temp_state, available_actions)

                        # reverse action back to Black's POV
                        black_action = self.env.reverse_action(best_action)

                    elif(self.env.opponent == 'random'):
                        black_actions = self.env.get_possible_actions()
                        black_action = random.choice(black_actions)
                    else:
                        raise ValueError("Invalid opponent type in environment")

                    new_state, black_move_reward, done, info = self.env.black_step(black_action)
                
                # update tables from whites persepctive
                if(done):
                    # when the new state is terminal there are no further actions from it and its Q value is 0
                    self.update_table(pre_w_state, white_action, w_move_reward+black_move_reward)
                else:
                    post_b_state = self.env.encode_state()
                    #make sure all actions are initialised in the lookup table
                    available_actions = self.env.possible_actions

                    best_action = self.best_action(post_b_state, available_actions)
                    self.update_table(pre_w_state, white_action, w_move_reward+black_move_reward, post_b_state, best_action)

                total_reward += w_move_reward+black_move_reward
                episode_length += 1

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

            epoch_reward.append(round(total_reward, 1))
            episode_lengths.append(episode_length)

        end = time.time()

        self.train_rewards = epoch_rewards
        self.test_rewards  = test_rewards
        self.train_lengths = epoch_episode_lengths
        self.test_lengths = test_lengths

        print("Training complete")
        print(f'Time taken: {round(end-start, 1)}')
        print(f"Number of epochs: {no_epochs}")
        print(f"Average episode length: {np.mean(episode_lengths)}")
        print(f"{len(self.Q)} states have been assigned values")
        print(f"Hyperparameters are: lr={self.lr}, discount={self.discount}, epsilon={self.epsilon}")

        if(save):
            self.save_training()

        self.show_rewards()
        
    
    def one_episode(self):
        environment = ChessEnvV2(player_color=self.env.player, opponent=self.env.opponent, log=False, initial_board=self.env.initial_board, end = self.env.end)
        total_reward = 0
        length = 0

        # iterate through moves
        while(not environment.done):
            length += 1
            available_actions = environment.possible_actions
            encoded_state = environment.encode_state()
            #make sure all actions are initialised in the lookup table

            action = self.best_action(encoded_state, available_actions)
            
            _, w_move_reward, _, _ = environment.white_step(action)
            total_reward += w_move_reward
            if(environment.done):
                break

            available_actions = environment.possible_actions
            
            action = random.choice(available_actions)

            _, black_reward, _, _ = environment.black_step(action)
            total_reward += black_reward
        
        return total_reward, length

    def update_table(self, state, action, reward, new_state=None, best_action=None):
        if new_state == None:
            ## new state is terminal so has a value of 0
            self.Q[(state, action)] += self.lr*(reward - self.Q[(state, action)])
        else:
            # update Q value
            self.Q[(state, action)] += self.lr*(reward + self.discount*self.Q[(new_state, best_action)] - self.Q[(state, action)])

    def choose_egreedy_action(self, state, actions):
        if(random.random() > self.epsilon):
            # select action with the largest value
            chosen_action = self.best_action(state, actions)
        else:
            # select random action
            chosen_action = random.choice(actions)
        
        return chosen_action
    
    def best_action(self, state, actions):
        # select action with the largest value
        values = [self.Q[(state, action)] for action in actions]
        max_value = max(values)
        # print(f"Best action value: {max_value}")

        # make sure that if there are multiple actions with the max value, one is chosen at random
        potential_actions = [actions[i] for i in range(len(actions)) if values[i]==max_value]
        best_action = random.choice(potential_actions)

        return best_action

    def play_human(self):
        print("Starting Game:")
        self.env.reset()
        total_reward = 0
        
        # iterate through moves
        while(not self.env.done):
            self.env.render()

            available_actions = self.env.possible_actions
            encoded_state = self.env.encode_state()
            action = self.best_action(encoded_state, available_actions)
            
            _, whtie_reward, _, _ = self.env.white_step(action)
            self.env.render()
            total_reward += whtie_reward
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
    
    def save_training(self):
        no_epochs = len(self.test_rewards)
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = self.name + " " + date_time + ", " + str(no_epochs) + " epochs"
        self.save_parameters(folder_name, "table.txt")
        self.save_rewards(folder_name, "rewards.txt")
        
    def save_rewards(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w') as f:
            f.write(f"{self.train_rewards}\n{self.test_rewards}\n{self.train_lengths}\n{self.test_lengths}")
        print(f"Reward logs have been written to '{filepath}' successfully.")
    
    def save_parameters(self, folder, filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Construct the full file path
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'w') as f:
            for key, value in self.Q.items():
                f.write(f"{key}: {value}\n")
        print(f"Dictionary elements have been written to '{filepath}' successfully.")

    def load_training(self, folder):
        # load model
        self.load_parameters(folder, "table.txt")

        # load rewards
        self.load_rewards(folder, "rewards.txt")

    def load_rewards(self, folder, filename):
        # Construct the full file path
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'r') as f:
            contents = f.read()
        parts = contents.split('\n')
        self.train_rewards = eval(parts[0])
        self.test_rewards = eval(parts[1])
        self.train_lengths = eval(parts[2])
        self.test_lengths = eval(parts[3])

        print(f"Rewards logs have been loaded from '{filepath}' successfully.")

    def load_parameters(self, folder, filename):
        # Construct the full file path
        filepath = os.path.join(folder, filename)
        
        with open(filepath, 'r') as f:
            # Read lines from the file
            lines = f.readlines()
            
            # Parse each line and extract key-value pairs
            for line in lines:
                key, value = line.strip().split(': ')
                self.Q[key] = float(value)

    def show_rewards(self):
        print("Showing rewards...")
        no_epochs = len(self.test_rewards)
        # calculate rolling averages
        window_size = 400
        average_test_rewards = [np.mean(self.test_rewards[i-window_size:i+1]) if i>window_size else max(0, np.mean(self.test_rewards[0:i+1])) for i in range(len(self.test_rewards))]
        average_rewards = [np.mean(self.train_rewards[i-window_size:i+1]) if i>window_size else np.mean(self.train_rewards[0:i+1]) for i in range(len(self.train_rewards))]
        plot_test_rewards(average_rewards, average_test_rewards)

    def show_lengths(self):
        print("Showing lengths...")
        no_epochs = len(self.test_rewards)
        # calculate rolling averages
        window_size = no_epochs//25
        average_test_lengths = [np.mean(self.test_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.test_lengths[0:i+1]) for i in range(len(self.test_lengths))]
        average_train_lengths = [np.mean(self.train_lengths[i-window_size:i+1]) if i>window_size else np.mean(self.train_lengths[0:i+1]) for i in range(len(self.train_lengths))]
        plot_episode_lengths(average_train_lengths, average_test_lengths)

    def get_0_proportion(self):
        count = 0
        for value in self.Q.values():
            if value == 0.0:
                count += 1
        
        print(f"Total: {len(self.Q)}, count: {count}")
        return count/len(self.Q)