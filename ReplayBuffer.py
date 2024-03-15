import numpy as np

class ReplayBuffer(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_index = 0

        self.state_memory = np.empty(self.memory_size, dtype='<U64')   # states are stored in the encoded string form (to save memory?)
        self.next_state_memory = np.empty(self.memory_size, dtype='<U64')
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store(self, state, next_state, action, reward, terminal):
        index = self.memory_index % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.memory_index += 1

    def sample_batch(self):
        index = min(self.memory_index, self.memory_size)
        batch = np.random.choice(index, self.batch_size, replace=False)

        state_batch = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]
        index_batch = np.arange(self.batch_size, dtype=np.int32)

        return dict(states=state_batch, next_states=next_state_batch, actions=action_batch, rewards=reward_batch, terminals=terminal_batch, indexes=index_batch)

    def full_batch(self):
        return self.memory_index < self.batch_size