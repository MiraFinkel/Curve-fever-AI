import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'legal_actions'))


class ExperienceReplay:
    def __init__(self, e_max=10000):
        self._max = e_max  # maximum number of experiences
        self.transitions = []  # total experiences the Agent stored

    def get_max(self):
        """return the maximum number of experiences"""
        return self._max

    def get_num(self):
        """return the current number of experiences"""
        return len(self.transitions)

    def reset(self):
        """resets the memory, deleting all previous experiences"""
        self.transitions = []

    def get_batch(self, batch_size: int):
        """randomly choose a batch of experiences for training"""
        if batch_size < self.get_num():  # We must sample with replacements
            batch = random.choices(self.transitions, k=batch_size)
        else:
            batch = random.sample(self.transitions, k=batch_size)
        batch = Transition(*zip(*batch))
        state = np.array(batch.state)
        action = np.array(batch.action)
        reward = np.array(batch.reward)
        next_state = np.array(batch.next_state)
        legal_actions = np.array(batch.legal_actions)
        return state, action, next_state, reward, legal_actions

    def add(self, state, action, reward, next_state, legal_actions):
        """remove the oldest experience if the memory is full"""
        if self.get_num() > self.get_max():
            del self.transitions[0]
        """add single experience"""
        self.transitions.append(Transition(state, action, reward, next_state, legal_actions))
