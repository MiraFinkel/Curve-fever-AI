from game.players.player import Player
import numpy as np


class DRLPlayer(Player):
    def __init__(self, player_id, game, model):
        super().__init__(player_id, game)
        self._net = model
        self.predictions = 0
        self.total_time = 0

    def get_action(self, state):
        # if self.predictions > 0 and not self.predictions % 100:
        #     print(f'average action takes {self.total_time / self.predictions} seconds')
        self.predictions += 1
        box = state.adjust_to_drl_player(self.id)  # self.crop_box(state.board, state.positions)
        # input = [box[np.newaxis, ...], np.array(angle)[np.newaxis, ...]]
        values = self._net.predict(box[np.newaxis, ...])
        return np.random.choice(np.flatnonzero(values == np.max(values)))

        # values = self._net.predict(box[np.newaxis, ...])
        # return np.random.choice(np.flatnonzero(values == np.max(values)))

        # legal_actions = self.game.legal_actions
        #
        # # Calculate the Q-value of each action
        # q_values = self._net.predict(box[np.newaxis, ...], np.expand_dims(legal_actions, 0))
        #
        # # Make sure we only choose between available actions
        # legal_actions = np.logical_and(legal_actions, q_values == np.max(q_values))
        #
        # return np.random.choice(np.flatnonzero(legal_actions))

