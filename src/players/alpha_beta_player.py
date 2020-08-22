import random

from src.players.player import Player
import numpy as np
import itertools
from static.settings import *
from src.environment.state import State
from typing import List


class AlphaBetaPlayer(Player):

    def __init__(self, player_id, game, depth: int):
        super().__init__(player_id, game)
        # print("Number of processors: ", mp.cpu_count())
        self.total_time = 0
        self.depth = depth  # the depth of the minmax (how many moves we look ahead)
        self.initial_opponents = []
        self.opponents = []
        self.n_of_opp = 0
        self.all_actions = []
        self.first_round = True
        # self.pool = mp.Pool(mp.cpu_count())
        self.states_evaluated = 0
        self.successors_generated = 0

    def get_action(self, state):
        state_copy = State.from_state(state)
        if 1 < len(state_copy.alive) < self.n_of_opp or self.first_round:
            self.update_opponents(state_copy)
        values = []
        initial_position = self.get_initial_positions([self.id], state_copy)
        for action in [RIGHT, LEFT, STRAIGHT]:
            self.successors_generated += 1
            self.update_successor_state([self.id], state_copy, [action], initial_position, True)
            values.append(self.alpha_beta(state_copy, self.depth, -np.inf, np.inf, False, action))
            self.update_successor_state([self.id], state_copy, [action], initial_position, False)

        max_vals = [i for i, val in enumerate(values) if val == max(values)]
        chosen_action = min(max_vals) if len(max_vals) <= 2 else random.choice(max_vals)

        return chosen_action

    def alpha_beta(self, state, depth, alpha, beta, max_player, potential_action):
        if depth == 0:
            fill_value, nearest_opponent = self.calc_fill_value(state)
            closest_obstacle_value = self.closest_obstacle_value(state, potential_action)
            weights = np.array([1, 0.2, 1])
            values = np.array([closest_obstacle_value, fill_value, nearest_opponent]).astype(int)
            tmp = np.multiply(values, weights)
            return np.sum(tmp)
        if max_player:  # the alpha_beta_player is max
            value = -np.inf
            initial_position = self.get_initial_positions([self.id], state)
            for action in [RIGHT, LEFT, STRAIGHT]:
                self.successors_generated += 1
                self.update_successor_state([self.id], state, [action], initial_position, True)
                value = max(value, self.alpha_beta(state, depth - 1, alpha, beta, False, action))
                self.update_successor_state([self.id], state, [action], initial_position, False)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:  # all of the other players are the opponent that is min
            value = self.closest_obstacle_value(state, potential_action)
            if len(self.all_actions) != 0:
                value = np.inf
                all_actions = random.sample(self.all_actions, 2)
                initial_position = self.get_initial_positions(self.opponents, state)
                for actions in all_actions:
                    self.successors_generated += 1
                    opponents = self.opponents
                    self.update_successor_state(opponents, state, actions, initial_position, True)
                    value = min(value, self.alpha_beta(state, depth - 1, alpha, beta, True, potential_action))
                    self.update_successor_state(opponents, state, actions, initial_position, False)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value

    def update_successor_state(self, player_ids: List[int], state: State, actions: List[int], initial_positions,
                               forward):
        for i, player in enumerate(player_ids):
            positions_to_erase = []
            position = initial_positions[i][0]
            angle = initial_positions[i][1]
            for _ in range(5):
                position, angle = self.game.update_pos_angle(position, angle, actions[i])
                if not forward:
                    positions_to_erase.append((position, angle))
                else:
                    state.set_position(player, position)
                    state.set_angle(player, angle)
                    if self.game.detect_collision(player, state):
                        state.alive[player] = False
                        if player != self.id:
                            self.update_opponents(state)
                        break
                    state.draw_player(player, True)

            # if not player_died or not forward:
            if not forward:
                if not state.alive[player]:
                    state.alive[player] = True
                    if player != self.id:
                        self.update_opponents(state)
                erase_head = True
                for pos, angle in positions_to_erase[::-1]:
                    if self.game.in_bounds(pos) and state.get_2d_pixel(pos) == (player + 2) or not erase_head:
                        if erase_head:
                            state.draw_circle(BLACK_2D, BLACK,
                                              self.game.get_head_position(pos, angle), HEAD_RADIUS)
                            erase_head = False
                        state.draw_circle(BLACK_2D, BLACK, pos, PLAYER_RADIUS)
                state.set_position(player, initial_positions[i][0])
                state.set_angle(player, initial_positions[i][1])
                state.draw_player(player)
            if state.alive[player]:
                head_position = self.game.get_head_position(state.get_position(player), state.get_angle(player))
                state.draw_head(head_position)

    def get_initial_positions(self, player_ids, state):
        initial_pos = []
        for player in player_ids:
            initial_pos.append((state.get_position(player), state.get_angle(player)))
        return initial_pos

    def closest_obstacle_value(self, state, action):
        self.states_evaluated += 1
        value = 0
        new_pos = state.get_position(self.id)
        new_angle = state.get_angle(self.id)
        if self.game.detect_collision(self.id, state) or not state.alive[self.id]:
            value -= 100
        for i in range(40, 0, -1):
            new_pos = self.game.calculate_new_position(new_pos, new_angle)
            head_pos = self.game.get_head_position(new_pos, new_angle)
            if self.game.detect_collision(self.id, state, head_pos):
                value -= 10 * i
                break
        return value

    def update_opponents(self, state):
        self.n_of_opp = len(state.alive)
        if self.first_round:
            self.initial_opponents = [i for i in range(self.n_of_opp) if i != self.id]
            self.first_round = False
        if self.n_of_opp > 1:
            self.opponents = [opponent for opponent in self.initial_opponents if state.alive[opponent]]
            self.all_actions = list(itertools.product(range(3), repeat=self.n_of_opp - 1))

    def calc_fill_value(self, state):
        ab_heuristic = AlphaBetaHeuristic(self, self.game, state, self.opponents)
        positions = state.get_all_positions()
        angles = state.get_all_angles()
        score, euclidean_distance = ab_heuristic.score_function(positions, angles)
        return score, euclidean_distance


NEGATIVE_SCORE = 1000 * -127
SUB_SAMPLE_FACTOR = 1
FILL_FACTOR = 20
NUM_ANGLES = 2
DIST_FACTOR = 2 * FILL_FACTOR
POS_FACTOR = 300

class AlphaBetaHeuristic:
    def __init__(self, player, game, state, opponents):
        self.player = player
        self.game = game
        self.state = state
        self.opponents = opponents

    def calc_distances(self, start, old_angle):
        start = int(round(start[0])), int(round(start[1]))
        distances = np.ones((int(ARENA_WIDTH / SUB_SAMPLE_FACTOR), int(ARENA_HEIGHT / SUB_SAMPLE_FACTOR))) * 100
        distances[start[0], start[1]] = 0
        front = {(start, old_angle)}

        flag = False

        while len(front):
            old_position, old_angle = front.pop()
            legal_moves = []
            for action in ACTIONS:
                new_angle = self.game.calculate_new_angle(old_angle, action)
                new_position = self.game.get_head_position(old_position, new_angle)
                new_position = int(round(new_position[0])), int(round(new_position[1]))
                if self.game.in_bounds(new_position) and self.state.is_2d_pos_available(new_position):
                    legal_moves.append((new_position, new_angle))

            for new_position, new_angle in legal_moves:
                if not self.game.in_bounds(new_position):
                    continue
                new_distance = distances[old_position[0], old_position[1]] + 1
                old_distance = distances[new_position[0], new_position[1]]
                if new_distance < old_distance:
                    distances[new_position[0], new_position[1]] = new_distance
                    if len(front) == FILL_FACTOR:
                        flag = True
                        break
                    front.add((new_position, new_angle))
            if flag:
                break
        return distances

    def score_function(self, positions, angles):
        max_player_pos = positions[self.player.id]
        max_player_angle = angles[self.player.id]
        opponents_angles = [angles[i] for i in self.opponents]
        opponents_pos, euclidean_distance = self.get_euclidean_dist_and_pos(max_player_angle, max_player_pos,
                                                                            positions)
        if len(opponents_pos) < 1:
            return 0, euclidean_distance

        # max_player_pos = int(max_player_pos[0] / SUB_SAMPLE_FACTOR), int(max_player_pos[1] / SUB_SAMPLE_FACTOR)
        # opponents_pos = [(int(pos[0] / SUB_SAMPLE_FACTOR), int(pos[1] / SUB_SAMPLE_FACTOR)) for pos in opponents_pos]
        # distances = np.ones(
        #     (int(ARENA_WIDTH / SUB_SAMPLE_FACTOR), int(ARENA_HEIGHT / SUB_SAMPLE_FACTOR), len(opponents_pos)))

        if self.game.detect_collision(self.player.id, self.state) or not self.state.alive[self.player.id]:
            return NEGATIVE_SCORE, euclidean_distance  # Return a very negative score

        nearest_tiles_max_player, nearest_tiles_opponents = self.get_distance_values(max_player_angle,
                                                                                     max_player_pos, opponents_angles,
                                                                                     opponents_pos)
        diff = nearest_tiles_max_player - nearest_tiles_opponents
        return diff, euclidean_distance

    def get_distance_values(self, max_player_angle, max_player_pos, opponents_angles, opponents_pos):
        distances = np.ones((ARENA_WIDTH, ARENA_HEIGHT, len(opponents_pos)))
        max_player_distances = np.ones((int(ARENA_WIDTH / SUB_SAMPLE_FACTOR), int(ARENA_HEIGHT / SUB_SAMPLE_FACTOR), 1))
        max_player_distances[:, :, 0] = self.calc_distances(max_player_pos, max_player_angle)
        for i in range(len(opponents_pos)):
            distances[:, :, i] = self.calc_distances(opponents_pos[i], opponents_angles[i])
        opponents_distances = np.amin(distances, axis=2)

        # Calculate difference in number of tiles the players can reach first
        nearest_tiles_max_player = np.sum(max_player_distances[:, :, 0] < opponents_distances[:, :])
        nearest_tiles_opponents = np.sum(max_player_distances[:, :, 0] > opponents_distances[:, :])
        return nearest_tiles_max_player, nearest_tiles_opponents

    def get_euclidean_dist_and_pos(self, max_player_angle, max_player_pos, positions):
        tmp_opponents_pos = [positions[i] for i in self.opponents]
        front_angles = calc_angles(max_player_angle)
        opponents_pos, euclidean_distances = [], []
        for pos in tmp_opponents_pos:
            dist = self.game.distance_between_two_pos(max_player_pos, pos)
            if is_pos_within_range(max_player_pos, max_player_angle, pos, front_angles[0], front_angles[1]):
                euclidean_distances.append(dist)
            if dist < DIST_FACTOR:
                opponents_pos.append(pos)
        euclidean_distances = np.array(euclidean_distances)
        euclidean_distance = POS_FACTOR if len(euclidean_distances) == 0 else np.amin(euclidean_distances, axis=0)

        return opponents_pos, euclidean_distance * 0.1


def calc_angles(initial_angle):
    angles = [(-np.pi / 4) + ((i / (NUM_ANGLES - 1)) * (np.pi / 2)) for i in range(NUM_ANGLES)]
    angles = initial_angle + np.array(angles)
    return angles


def is_pos_within_range(max_player_pos, max_player_angle, pos, min_angle, max_angle):
    pos_x, pos_y = pos[0] - max_player_pos[0], pos[1] - max_player_pos[1]
    max_x, max_y = (0, 0)
    n = max_y - (np.tan(max_player_angle) * max_x)
    x = max_x + 2
    y = np.tan(max_player_angle) * x + n
    angle_to_pos = np.arctan2((pos_y, y), (pos_x, x))
    angle_to_pos = np.abs(angle_to_pos[1] - angle_to_pos[0])
    return (angle_to_pos > min_angle) and (angle_to_pos < max_angle)
