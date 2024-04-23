import numpy as np

class TradjectoryMemory():
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.state_memory = []
        self.action_memory = []
        self.action_mask_memory = []
        self.reward_memory = []
        self.value_memory = []
        self.prob_memory = []
        self.terminal_memory = []

    def clear(self):
        self.state_memory = []
        self.action_memory = []
        self.action_mask_memory = []
        self.reward_memory = []
        self.value_memory = []
        self.prob_memory = []
        self.terminal_memory = []

    def store(self, state, action, available_actions, reward, value, prob, terminal):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.action_mask_memory.append(available_actions)
        self.reward_memory.append(reward)
        self.value_memory.append(value)
        self.prob_memory.append(prob)
        self.terminal_memory.append(terminal)

    def get_batches(self):
        n_states = len(self.state_memory)
        n_batches = n_states // self.batch_size
        indicies = np.arange(n_states)
        np.random.shuffle(indicies)
        batches =  [indicies[i * self.batch_size:(i+1) * self.batch_size] for i in range(n_batches)]

        return batches
    
    def get_memory(self):
        return np.array(self.state_memory), np.array(self.action_memory), np.array(self.action_mask_memory), np.array(self.reward_memory), np.array(self.value_memory), np.array(self.prob_memory), np.array(self.terminal_memory)
    
    def full_batch(self):
        return len(self.state_memory) >= self.batch_size