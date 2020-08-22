import abc


class DQNEnv:

    def __init__(self):
        self.state = None

    message = 'Please implement this method'

    @abc.abstractmethod
    def reset(self):
        """ Resets the environment to an initial state """
        raise NotImplementedError(DQNEnv.message)

    @abc.abstractmethod
    def step(self, action):
        """ Returns the next state and the reward achieved from taking specified action from self.state
            If achieved state is a terminal state, returns None as the next_state """
        raise NotImplementedError(DQNEnv.message)

    def get_state(self):
        return self.state

    @abc.abstractmethod
    def get_legal_actions(self, state):
        """ Returns a boolean vector representing the legal actions that can be taken from specified state"""
        raise NotImplementedError(DQNEnv.message)

    # @abc.abstractmethod
    # def is_terminal_state(self, state):
    #     """ Returns True iff specified state is a terminal state"""
    #     raise NotImplementedError(DQNEnv.message)
