import numpy as np
import abc
from static.settings import *
import copy


class State:
    count = 0

    def __init__(self, shape, positions, angles, colors):
        self.margin = 6
        # shape = shape[0] + (self.margin * 2), shape[1] + (self.margin * 2), shape[2]
        # self._rgb_board[:self.margin - 8, :, ...] = BLACK
        # self._rgb_board[ARENA_HEIGHT + self.margin + 8:, :, ...] = BLACK
        # self._rgb_board[:, :self.margin - 8, ...] = BLACK
        # self._rgb_board[:, ARENA_WIDTH + self.margin + 8:, ...] = BLACK

        shape_3d = shape[0] + (self.margin * 2), shape[1] + (self.margin * 2), shape[2]
        shape_2d = (shape[0], shape[1])
        self._rgb_board = np.ones(shape_3d) * WHITE
        self._board = np.ones(shape_2d) * WHITE_2D
        self.reset_arena()
        self.colors = colors
        self._positions = positions
        self.alive = [True for _ in range(len(positions))]
        self._angles = angles
        self.counts = [0 for _ in angles]

    def get_3d_pixel(self, coord):
        return self._rgb_board[self.margin + int(round(coord[1])), self.margin + int(round(coord[0])), ...]

    def get_2d_pixel(self, coord):
        return self._board[int(round(coord[1])), int(round(coord[0]))]

    def _get_rgb_board(self):
        return self._rgb_board

    def get_board(self):
        return self._board

    def is_2d_pos_available(self, coord):
        pixel = self.get_2d_pixel(coord)
        return pixel == 0

    # def set_board(self, board: np.ndarray):
    #     arena = self.get_rgb_board()
    #     arena[...] = board.copy()

    def get_position(self, player_id):
        pos = self._positions[player_id]
        return pos

    def get_all_positions(self):
        return self._positions

    def get_all_angles(self):
        return self._angles

    def set_position(self, player_id, position):
        self._positions[player_id] = position

    def adjust_pos_to_board_with_margin(self, position):
        return position[0] + self.margin, position[1] + self.margin

    def get_angle(self, player_id):
        return self._angles[player_id]

    def set_angle(self, player_id, angle):
        self._angles[player_id] = angle

    def reset_arena(self):
        self._rgb_board[self.margin:self.margin + ARENA_HEIGHT, self.margin:self.margin + ARENA_WIDTH, ...] = BLACK
        self._board[: ARENA_HEIGHT, : ARENA_WIDTH] = BLACK_2D

    def adjust_to_drl_player_no_position(self, player_id):
        features = self.adjust_to_drl_player(player_id)
        return features[:-2]

    def adjust_to_drl_player(self, player_id):
        ## Feature representation
        max_dist = 350
        num_angles = 25
        return self.get_distance_to_obstacles(self._angles[player_id], self._positions[player_id], num_angles, max_dist)

    @abc.abstractmethod
    def update_player_graphics(self, player_id):
        raise NotImplementedError('Please implement this method')

    @classmethod
    def from_state(cls, other, *args, **kwargs):
        # return copy.deepcopy(other)
        new_state = copy.copy(other)
        new_state.margin = other.margin
        new_state._rgb_board = other._get_rgb_board().copy()
        new_state._board = other.get_board().copy()
        new_state.colors = other.colors
        new_state._positions = copy.deepcopy(other.get_all_positions())
        new_state.alive = copy.copy(other.alive)
        new_state._angles = copy.copy(other.get_all_angles())
        return new_state

    def draw_circle(self, color_2d, color, center, radius):
        circle = CIRCLES[radius - 1]
        circle = np.array(circle) + np.array(center)
        circle = np.round(circle).astype(np.int)
        circle = self.clip(circle)
        self._rgb_board[circle[..., 1] + self.margin, circle[..., 0] + self.margin, ...] = color
        self._board[circle[..., 1], circle[..., 0]] = color_2d

    def clip(self, circle):
        circle[circle < 0] = 0
        circle[circle[..., 0] >= ARENA_WIDTH, 0] = ARENA_WIDTH - 10
        circle[circle[..., 1] >= ARENA_HEIGHT, 1] = ARENA_HEIGHT - 10
        return np.round(circle).astype(np.int)

    def draw_player(self, player_id, use_color=False):
        self.counts[player_id] += 1
        rgb_color = self.colors[player_id] if use_color else BLACK
        color = player_id + 2 if use_color else BLACK_2D
        try:
            self.draw_circle(color, rgb_color, self._positions[player_id], PLAYER_RADIUS)
        except:
            self.draw_circle(color, rgb_color, self._positions[player_id], PLAYER_RADIUS)

    def draw_head(self, position):
        try:
            self.draw_circle(HEAD_2D, HEAD_COLOR, position, HEAD_RADIUS)
        except:
            self.draw_circle(HEAD_2D, HEAD_COLOR, position, HEAD_RADIUS)

    def is_terminal_state(self):
        return np.sum(self.alive) < 1

    def get_distance_to_obstacles(self, initial_angle, position, num_angles, max_distance=150):
        angles = [(-np.pi / 2) + ((i / (num_angles - 1)) * np.pi) for i in range(num_angles)]
        angles = initial_angle + np.array(angles)
        distances = np.zeros_like(angles)
        for i, angle in enumerate(angles):
            distances[i] = self.distance_to_obstacle(position, angle, max_distance)
        return distances / max_distance

    def distance_to_obstacle(self, position, angle, max_distance=150):
        position = np.array(position) + self.margin
        distances = np.arange(10, max_distance)
        xx = position[0] + (np.cos(angle) * distances)
        yy = position[1] - (np.sin(angle) * distances)
        xx = np.round(xx).astype(np.int)
        yy = np.round(yy).astype(np.int)
        xx[xx < self.margin] = self.margin - 1
        yy[yy < self.margin] = self.margin - 1
        xx[xx >= ARENA_WIDTH + (self.margin * 2)] = ARENA_WIDTH + (self.margin * 2) - 1
        yy[yy >= ARENA_HEIGHT + (self.margin * 2)] = ARENA_HEIGHT + (self.margin * 2) - 1
        nonzero = np.nonzero(self._rgb_board[yy, xx, ...])
        if len(nonzero[0]) > 0:
            return np.linalg.norm(position - np.array([xx[nonzero[0][0]], yy[nonzero[0][0]]]))
        return max_distance

    def get_player_drl_features(self, player_id):
        features = [self._positions[player_id][0] / ARENA_WIDTH, self._positions[player_id][1] / ARENA_HEIGHT]
        return features
