import random, time, copy, os
import numpy as np
import gym
from gym_chess import ChessEnvV1, ChessEnvV2
from q_learning import Q_learning_agent

class Sarsa_lambda_agent(Q_learning_agent):
    def __init__(self, environment, alpha=0.2, discount=1.0, epsilon=0.15, trace_decay=0.7):
        super().__init__(environment, alpha=alpha, discount=discount, epsilon=epsilon)

        # set remaining hyperparameters
        self.trace_decay = trace_decay

        # create trace dictionary
        self.e = {}

    def train(self, no_episodes=0):
        episode_rewards = []
        test_rewards = []
        episode_lengths = []

        print('Starting position:')
        self.env.render()
        print("Training...")
        start = time.time()

        # loop for each episode
        for i in range(no_episodes):
            print(i)
            total_reward = 0
            self.env.reset()

            episode_length = 0
            pre_b_state = None

            # loop for each action in an episode
            done = False
            while(not done):
                pre_w_state = self.env.encode_state()

                # White's move
                available_actions = self.env.possible_actions
                # initialise actions in the lookup table
                self.initialise_values(pre_w_state, available_actions)

                white_action = self.choose_egreedy_action(pre_w_state, available_actions)
                new_state, w_move_reward, done, info = self.env.white_step(white_action)

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
                        self.initialise_values(encoded_temp_state, available_actions)
                        
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
                    self.initialise_values(post_b_state, available_actions)

                    # new_action = self.best_action(post_b_state, available_actions)
                    new_action = self.choose_egreedy_action(post_b_state, available_actions)
                    self.update_table(pre_w_state, white_action, w_move_reward+black_move_reward, post_b_state, new_action)

                total_reward += w_move_reward+black_move_reward
                episode_length += 1

            episode_rewards.append(round(total_reward, 1))
            episode_lengths.append(episode_length)
            test_rewards.append(self.one_episode())

        end = time.time()

        # Create an array to store the rolling averages
        average_rewards = np.zeros_like(episode_rewards, dtype=float)
        average_test_rewards = np.zeros_like(test_rewards, dtype=float)

        # Calculate the rolling averages
        over = no_episodes//50
        if(over-1>10):
            for i in range(10, over-1):
                average_rewards[i] = np.mean(episode_rewards[0:i])
                average_test_rewards[i] = np.mean(test_rewards[0:i])

        for i in range(over-1, len(episode_rewards)):
            average_rewards[i] = np.mean(episode_rewards[i+1-over:i+1])
            average_test_rewards[i] = np.mean(test_rewards[i+1-over:i+1])

        print("Training complete")
        print(f'Time taken: {round(end-start, 1)}')
        print(f"Number of episodes: {no_episodes}")
        print(f"Average episode length: {np.mean(episode_lengths)}")
        print(f"{len(self.Q)} states have been assigned values")
        print(f"Hyperparameters are: alpha={self.alpha}, discount={self.discount}, epsilon={self.epsilon}, trace_decay: {self.trace_decay}")
        
        return(average_rewards, average_test_rewards)


    def initialise_values(self, encoded_state, available_actions):
        for option in available_actions:
            if((encoded_state, option) not in self.Q):
                self.Q[(encoded_state, option)] = self.default_value  #initialise all action value pairs as zero
                self.e[(encoded_state, option)] = 0.0

    def update_table(self, state, action, reward, new_state=None, new_action=None):
        if new_state == None:
            # if the new state is terminal, it have a value of 0
            delta = reward - self.Q[(state, action)]
        else:
            delta = reward + self.discount*self.Q[(new_state, new_action)] - self.Q[(state, action)]

        self.e[(state, action)] += 1

        # Convert dictionary values to a NumPy array
        Q_values_array = np.array(list(self.Q.values()))
        e_values_array = np.array(list(self.e.values()))
        
        # Perform element-wise addition/multiplication
        Q_values_array += self.alpha*delta*e_values_array
        e_values_array *= self.discount*self.trace_decay
        
        # Convert the result back to a dictionary
        self.Q = {key: value for key, value in zip(self.Q.keys(), Q_values_array)}
        self.e = {key: value for key, value in zip(self.e.keys(), e_values_array)}
    
