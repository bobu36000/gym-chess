import random
import numpy as np
# OpenAI segment tree
from segment_tree import MinSegmentTree, SumSegmentTree

class ReplayBuffer(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_index = 0

        self.state_memory = np.empty(self.memory_size, dtype=dict)   # states are stored in the normal dictionary form
        self.next_state_memory = np.empty(self.memory_size, dtype=dict)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.next_available_actions_memory = np.empty(self.memory_size, dtype=np.ndarray)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store(self, state, next_state, action, next_available_actions, reward, terminal):
        index = self.memory_index % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.next_available_actions_memory[index] = next_available_actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.memory_index += 1

    def sample_batch(self):
        index = min(self.memory_index, self.memory_size)
        batch = np.random.choice(index, self.batch_size, replace=False)

        state_batch = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch = self.action_memory[batch]
        next_available_actions_batch = self.next_available_actions_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]
        index_batch = np.arange(self.batch_size, dtype=np.int32)

        return dict(states=state_batch, next_states=next_state_batch, actions=action_batch, next_available_actions=next_available_actions_batch, rewards=reward_batch, terminals=terminal_batch, indexes=index_batch)

    def full_batch(self):
        return self.memory_index >= self.batch_size
    

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, memory_size, batch_size, alpha):
        super().__init__(memory_size, batch_size)

        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_index = 0

        capacity = 1
        while capacity < self.memory_size:
            capacity *= 2

        self.min_tree = MinSegmentTree(capacity)
        self.sum_tree = SumSegmentTree(capacity)

    def store(self, state, next_state, action, next_available_actions, reward, terminal):
        super().store(state, next_state, action, next_available_actions, reward, terminal)

        index = self.tree_index % self.memory_size
        self.min_tree[index] = self.max_priority ** self.alpha
        self.sum_tree[index] = self.max_priority ** self.alpha
        self.tree_index += 1

    def sample_batch(self, beta):
        batch = self.choice_batch()

        state_batch = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]
        weight_batch = np.array(self.weigh_batch(batch, beta), dtype=np.float32)
        batch_batch = np.array(batch, dtype=np.int32)
        index_batch = np.arange(self.batch_size, dtype=np.int32)

        return dict(states=state_batch, next_states=next_state_batch, actions=action_batch, rewards=reward_batch, terminals=terminal_batch, weights=weight_batch, batches=batch_batch, indexes=index_batch)
    
    def choice_batch(self):
        index = min(self.memory_index, self.memory_size)
        len_priority = self.sum_tree.sum(0, index-1) / self.batch_size

        batch = []
        for i in range(self.batch_size):
            upperbound = random.uniform(len_priority * i, len_priority * (i+1))
            batch.append(self.sum_tree.find_prefixsum_idx(upperbound))

        return batch

    def weigh_batch(self, batch, beta):
        index = min(self.memory_index, self.memory_size)
        sum_priority = self.sum_tree.sum()
        weight_max = ((self.min_tree.min() / sum_priority) * index) ** (-beta)
        weights = []
        for i in batch:
            weight = ((self.sum_tree[i] / sum_priority) * index) ** (-beta)
            weights.append(weight / weight_max)
        return weights
    
    def update_batch(self, batch, priorities):
        for i, priority in zip(batch, priorities):
            self.min_tree[i] = priority ** self.alpha
            self.sum_tree[i] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def full_batch(self):
        return super().full_batch()