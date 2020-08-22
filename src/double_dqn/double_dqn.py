from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import clone_model, load_model, model_from_json
import numpy as np


class DoubleDQN:
    def __init__(self, model, discount: float = 0.95):
        # , optimizer='adam', loss='mse', discount: float = 0.95):
        """
        C-tor
        """
        self.discount = discount
        self.q_net = None
        self.target_net = None
        self.set_model(model)

    def align_target_model(self):
        """
        Sets the target net weights to be the same as the q-net.
        """
        self.target_net.set_weights(self.q_net.get_weights())

    def predict(self, states: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
        """
        Given a state and legal actions that can be taken from that step, calculates the Q-values of (state, action) for
        each action that can be taken (illegal actions are evaluated as 0). Also works for a batch of states.
        :param states: a batch of states
        :param legal_actions: a batch of boolean vectors representing the legal actions from each step.
        :return: A numpy.ndarray representing the Estimated Q-values of all actions that can be taken from the specified
                state
        """

        q_values = self.q_net.predict(states)
        illegal = np.where(np.logical_not(legal_actions))
        q_values[illegal[0], illegal[1]] = 0  # setting q_values of illegal actions to 0. TODO: Check maybe set to -inf
        return q_values

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
            legal_actions: np.ndarray):
        """
        Updates the net according to the Double Q-learning paradigm.
        :param states: A batch of states.
        :param actions: A batch of actions taken from the specified states.
        :param next_states: the observed next-states after taking the specified actions from the specified states
                (Expects None if the state was a terminal state).
        :param rewards: A batch of rewards given after taking soecified actions from specified states and transitioning
                to specified next_states.
        :param legal_actions: A batch of legal actions that are allowed from the specified states.
        """
        if self.q_net is None:
            raise NotImplementedError('model was not initiated')
        targets = self.predict(states, legal_actions)
        hist = np.histogram(np.argmax(targets, axis=1), bins=[-0.5, 0.5, 1.5, 2.5], density=True)[0]
        # Create masks for separating terminal states from non terminal states.
        terminal_mask = np.array([next_state is None for next_state in list(next_states)])
        non_terminal_mask = np.logical_not(terminal_mask)
        terminal_mask = np.where(terminal_mask)[0]
        non_terminal_mask = np.where(non_terminal_mask)[0]

        # Update the expected sum of rewards in the terminal states to be just the current reward
        if terminal_mask.size > 0:
            targets[terminal_mask, actions[terminal_mask]] = rewards[terminal_mask]

        # Calculate the expected sum of rewards based on the target net and q net
        t = self.target_net.predict(np.array(list(next_states[non_terminal_mask])))
        q = self.q_net.predict(np.array(list(next_states[non_terminal_mask])))

        # Double-Q learning paradigm. We choose the actions based on the q_net evaluation, but evaluate those chosen
        # actions using the target net
        max_actions = np.argmax(q, axis=-1)
        estimated_values = t[np.arange(t.shape[0]), max_actions]
        targets[non_terminal_mask, actions[non_terminal_mask]] = np.add(rewards[non_terminal_mask],
                                                                        self.discount * estimated_values)

        # At this point the target is similar to the q-net prediction, except in the index corresponding to action
        # taken. In this index, the target value is just the reward if state is terminal, otherwise it is
        # reward + discount * Q(next_state, action), where Q(next_state, action) is evaluated using the Double
        # Q-learning algorithm. that is Q(next_state, action) = t_net(next_state)[argmax(q_net(next_state))]

        self.q_net.fit(states, targets, epochs=10, verbose=0)

    def set_model(self, model):
        self.q_net = model
        self.target_net = clone_model(self.q_net)
        self.target_net.set_weights(self.q_net.get_weights())

    # This handles saving\loading the model as explained here:
    # https://www.tensorflow.org/guide/keras/save_and_serialize (Ctrl+Left_click to open)

    def load_weights(self, path):
        self.q_net.load_weights(path)
        self.target_net = clone_model(self.q_net)

    def save_weights(self, path):
        self.q_net.save_weights(path)

    def to_json(self, **kwargs):
        return self.q_net.to_json(**kwargs)

    def from_json(self, json_config):
        self.set_model(model_from_json(json_config))

    def save_model(self, path):
        self.q_net.save(path)

    def load_model(self, path):
        self.set_model(load_model(path))
