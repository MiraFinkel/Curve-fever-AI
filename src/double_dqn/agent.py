import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from double_dqn.double_dqn import DoubleDQN
from double_dqn.experience_replay import ExperienceReplay
from tensorflow.keras.models import load_model, model_from_json
from matplotlib import pyplot as plt


class DQNAgent:
    def __init__(self, env, net_update_rate: int = 25, exploration_rate: float = 1.0,
                 exploration_decay: float = 0.000001):
        # set hyper parameters
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.net_updating_rate = net_update_rate

        # set environment
        self.env = env
        self.state_shape = env.get_state().shape
        self.action_shape = self.env.get_legal_actions(self.env.get_state()).shape

        # the number of experience per batch for batch learning
        # Experience Replay for batch learning
        self.exp_rep = ExperienceReplay()

        # Deep Q Network
        self.net = None

    def get_action(self, state: np.ndarray, eps) -> int:
        """Given a state returns a random action with probability eps, and argmax(q_net(state)) with probability 1-eps.
           (only legal actions are considered)"""

        legal_actions = self.env.get_legal_actions(state)
        if np.random.random() >= eps:  # Exploitation

            # Calculate the Q-value of each action
            q_values = self.net.predict(state[np.newaxis, ...], np.expand_dims(legal_actions, 0))

            # Make sure we only choose between available actions
            legal_actions = np.logical_and(legal_actions, q_values == np.max(q_values))

        return np.random.choice(np.flatnonzero(legal_actions))

    def update_net(self, batch_size: int):
        """ if there are more than batch_size experiences, Optimizes the network's weights using the Double-Q-learning
         algorithm with a batch of experiences, else returns"""
        if self.exp_rep.get_num() < batch_size:
            return
        batch = self.exp_rep.get_batch(batch_size)
        self.net.fit(*batch)

    def train(self, episodes: int, weight_path, checkpoint_path, max_actions: int = None, batch_size: int = 64,
              checkpoint_rate=100):
        """
        Runs a training session for the agent
        :param episodes: number of episodes to train.
        :param max_actions: max number of steps in an episode. if 0, each episode runs until reaching a terminal state.
        :param batch_size: number of experiences to learn from in each net_update.
        :return A tuple containing 2 lists. the first is a list of accumulated rewards for each training episode.
                The second list contains the number of actions taken in each training episode.
        """
        if self.net is None:
            raise NotImplementedError('agent.train called before model was not initiated. Please set the agent\'s model'
                                      ' using the set_model method. You can access the state and action shapes using '
                                      'agent\'s methods \'get_state_shape\' and \'get_action_shape\'')

        # set hyper parameters
        exploration_rate = self.exploration_rate
        total_rewards, average_reward_over_100 = [], []
        num_actions = []
        # start training
        for i in tqdm(range(episodes)):
            self.env.reset()  # Reset the environment for a new episode
            state = self.env.get_state()  # Get starting state
            step = 0
            ep_reward = 0
            while max_actions is None or step <= max_actions:

                step += 1
                action = self.get_action(state, exploration_rate)
                next_state, reward = self.env.step(action)

                ep_reward += reward
                # Add experience to memory
                self.exp_rep.add(state, action, reward, next_state, self.env.get_legal_actions(state))
                self.update_net(batch_size)  # Optimize the DoubleQ-net
                # if not step % 20:
                if next_state is None:  # The action taken led to a  terminal state
                    break
                if (step % self.net_updating_rate) == 0:
                    # update target network
                    self.net.align_target_model()

                state = next_state

            # Update total_rewards and num_actions to keep track of progress
            total_rewards.append(ep_reward)
            num_actions.append(step)
            # Update target network at the end of the episode
            self.net.align_target_model()
            if self.exp_rep.get_num() > batch_size:
                if (i % 50) == 0 and i >= 100:
                    average_reward_over_100.append(np.mean(total_rewards[-100:]))
                if (i % checkpoint_rate) == 0:
                    self.save_weights(weight_path + os.path.sep + f'episode_{i}_weights')
                    with open(checkpoint_path + os.path.sep + 'total_rewards.csv', 'w') as reward_file:
                        rewards = pd.DataFrame(total_rewards)
                        rewards.to_csv(reward_file)
                    with open(checkpoint_path + os.path.sep + 'exploration_rate', 'w') as exp_rate:
                        exp_rate.writelines([str(exploration_rate)])
                    plt.plot(np.arange(len(average_reward_over_100)), average_reward_over_100)
                    plt.title('average reward over last 100 episodes')
                    plt.xticks = 10 + (np.arange(len(average_reward_over_100)) * 5)
                    plt.xlabel('episodes / 10')
                    plt.ylabel('average reward over last 100 episodes')
                    plt.savefig(checkpoint_path + 'average_100')
                    plt.show()

            # Update exploration rate
            exploration_rate = max(0.1, 0.01 + (exploration_rate - 0.01) * np.exp(-self.exploration_decay * (i + 1)))
        self.exploration_rate = exploration_rate
        return total_rewards, num_actions

    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    def set_model(self, model):
        """ """
        self.net = DoubleDQN(model)

    # Handles saving\loading the model as explained here: https://www.tensorflow.org/guide/keras/save_and_serialize
    def load_weights(self, path):
        self.net.load_weights(path)

    def save_weights(self, path):
        self.net.save_weights(path)

    def save_model(self, path):
        if self.net is None:
            raise NotImplementedError('agent.save_model was called before model was not initiated. Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        self.net.save_model(path)

    def load_model(self, path):
        model = load_model(path)
        self.set_model(model)

    def to_json(self, **kwargs):
        if self.net is None:
            raise NotImplementedError('agent.to_json was called before model was not initiated. Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        return self.net.to_json(**kwargs)

    def from_json(self, json_config):
        model = model_from_json(json_config)
        self.set_model(model)
