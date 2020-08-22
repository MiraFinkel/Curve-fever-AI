import random
from game.players.player import Player
import numpy as np
import itertools


class AlphaBetaPlayer(Player):

    def __init__(self, player_id, game, depth: int):
        super().__init__(player_id, game)
        self.total_time = 0
        self.depth = depth  # the depth of the minmax (how many moves we look ahead)

    def get_action(self, state):
        # Choose one of the best actions
        values = []
        actions = self.game.get_legal_actions(state, self.id)
        for action in actions:
            successor = self.game.get_next_state(self.id, state, action)
            values.append(self.alpha_beta(successor, self.depth, -np.inf, np.inf, False))
        return actions[np.argmax(values)]

    def alpha_beta(self, state, depth, alpha, beta, max_player):
        # TODO: decide what is an 'end' state - right now it's when ab_player dies
        if self.game.detect_collision(self.id, state) or depth == 0:
            return self.evaluation_function(state)
        if max_player:  # the alpha_beta_player is max
            value = -np.inf
            for action in self.game.get_legal_actions(state, self.id):
                successor = self.game.get_next_state(self.id, state, action)
                value = max(value, self.alpha_beta(successor, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:  # all of the other players are the opponent that is min
            value = np.inf
            for successor in self.get_successor_states(state):
                value = min(value, self.alpha_beta(successor, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def get_successor_states(self, state):
        successors = []
        # the opponents that are alive
        opponents = [i for i, player in enumerate(self.game.players) if self.game.alive[i] and i != self.id]

        # here we shuffle them so that each time a different player starts
        random.shuffle(opponents)
        all_actions = list(itertools.product(range(3), repeat=len(opponents)))

        # TODO: decide how to go over all possible actions
        # here we sample 3 sets of actions to cut down running time, we may sample more - need to decide
        all_actions = random.sample(all_actions, 3)
        for actions in all_actions:
            successor = state
            for i, action in enumerate(actions):
                successor = self.game.get_next_state(opponents[i], successor, action)
            successors.append(successor)
        return successors

    def evaluation_function(self, state):
        return random.random()

