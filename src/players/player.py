import random
import abc
import numpy as np


class Player(object):
    speed = 4
    radius = 4
    d_theta = 0.09
    no_draw_time = 10

    def __init__(self, player_id, game):
        self.id = player_id
        self.game = game
        # self.head_color = GREEN
        # self.head_radius = 2
        # self.x = x
        # self.y = y
        # self.color = color
        # self.radius = Player.radius
        # self.initial_angle = angle
        # self.angle = angle
        # self.head = self.x, self.y
        # self.speed = Player.speed
        # self.d_theta = Player.d_theta
        # self.is_alive = True
        # self.draw_counter = 0
        # self.no_draw_counter = 0
        # self.draw_counter_limit = random.randint(100, 300)
        # self.no_draw_time = 10

    @abc.abstractmethod
    def get_action(self, state):
        raise NotImplementedError('Please implement this method ')

    # def act(self, action):
    #     if action == RIGHT:
    #         self.angle += self.d_theta
    #     if action == LEFT:
    #         self.angle -= self.d_theta
    #
    #     dx = np.cos(self.angle) * self.speed
    #     dy = np.sin(self.angle) * self.speed
    #     self.update_position(dx, dy)
    #
    # def set_position(self, x: int, y: int):
    #     self.x = int(x)
    #     self.y = int(y)
    #     hx = np.cos(self.angle) * (self.radius)
    #     hy = np.sin(self.angle) * (self.radius)
    #     self.head = int(hx + x), int(hy + y)
    #
    # def update_position(self, dx, dy):
    #     self.set_position(self.x + int(dx), self.y + int(dy))
    #
    # def get_current_color(self):
    #     if self.draw_counter >= self.draw_counter_limit:
    #         return BLACK
    #     return self.color
    #
    # def update_drawing_counters(self):
    #     if self.draw_counter >= self.draw_counter_limit:
    #         self.no_draw_counter += 1
    #         if self.no_draw_counter >= self.no_draw_time:
    #             self.draw_counter = 0
    #             self.no_draw_counter = 0
    #             self.draw_counter_limit = random.randint(100, 300)
    #             return
    #     self.draw_counter += 1
    #
    # def get_position(self):
    #     return self.x, self.y

    # def reset(self):
    #     self.is_alive = True
    #     self.set_position(random.randint(100, ARENA_WIDTH - 100),
    #                       random.randint(100, ARENA_HEIGHT - 100))
    #     self.angle = self.initial_angle
    #     self.draw_counter = 0
    #     self.no_draw_counter = 0
