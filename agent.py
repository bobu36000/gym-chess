from gym_chess import ChessEnvV1, ChessEnvV2

class Agent(object):
    def __init__(self, environment):
        self.env = environment

    def train(self, no_epochs):
        pass

    def one_episode(self):
        pass

    def choose_egreedy_action(self, state, actions):
        pass

    def best_action(self, state, actions):
        pass

    def play_human(self):
        pass

    def save_parameters(self):
        pass

    def load_parameters(self):
        pass